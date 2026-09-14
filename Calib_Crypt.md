# Calibration and the encrypted run: the box, the data, and what makes them fast

Everything this line of work needs that is not the algorithm: which machine
it runs on and what that machine actually gives you, what has to be
installed before anything is measured, which corpus plays which role, and
the loop that decides a calibration is good BEFORE a GPU hour is spent on
it. Written 2026-09-14 from a session that rebuilt the whole thing from an
empty overlay.

Companion to `BERT_BASE.md`, which is the design and the measurements. This
file is the operations half.

---

## 1. The box, and the number it lies about

`ssh vessl` — 1x A100-SXM4-80GB, Ubuntu 24.04, CUDA 13.0, 1121 GB RAM.

**The container has ELEVEN cores, not 96.**

```
$ nproc                      96          <- the host's
$ cat /sys/fs/cgroup/cpu.max
1100000 100000                           <- 1.1 s of CPU per 1.0 s = 11 cores
```

`nproc`, `os.sched_getaffinity`, `/sys/fs/cgroup/cpuset.cpus.effective` and
`vmstat`'s idle column all report the HOST's 96, so a job that is being
throttled looks exactly like a job that is blocked on something else:
`vmstat` will cheerfully say 83 % idle with a run queue of 112 while every
worker sits at 20 % of one core. **Read `cpu.max` before sizing any pool.**
Two hours of this session went into parallel plans built on 96.

Consequences that follow from 11:

* One BLAS thread per worker, a worker per core. OpenBLAS spin-waits on its
  barriers, so 24 processes x 4 threads thrash rather than share — measured
  at **260,000 context switches a second against 8.9 of 96 cores used**.
  With `OMP_NUM_THREADS=1` and ~9 workers it is 8,000 a second and the cores
  are actually busy.
* A crypto run is NOT purely a GPU job. Its kernel launches and staging are
  host work, and on 11 cores a scan beside it is expensive: the same layer
  with the same 3,840 bootstraps took **349.6 s with 24 scan workers and
  219.7 s without**. Scans take a `flock` and run `nice -n 19`.
* The overlay is wiped whenever the laptop's session drops. Anything small
  and expensive to remake belongs in git: **the calibrations do**
  (`bert_base/calib/T*.json`, 1.7–2.7 MB each, an hour of the box each).
  The previous session lost all of them and could not diff a regression
  against them.

## 2. What has to be installed, and what each one bought

The image ships none of these. In `bert_base/vessl_data.sh` and
`vessl_setup.sh`, all idempotent.

| what | why | measured |
|---|---|---|
| `libopenblas-dev` + `update-alternatives` | the image's numpy links the reference BLAS | **2 → 140 GFLOPS** |
| **`python3-scipy`** (apt, NOT pip) | `model.py`'s GELU falls back to `np.vectorize(math.erf)` — a **Python call per element**, 4.7 M of them per prompt | erf **48x**; the whole chain **1.9x** |
| `python3-pyarrow` (pip) | WikiText-2 ships as parquet | — |
| `python3.12-venv` then a venv with `numpy` `scipy` **`cupy-cuda13x`** | the scan AND the calibration on the card | scan **64 min → 8 min**; a T = 512 calibration **10 h → 42 s** |

Traps, each of which cost a cycle here:

* `pip install scipy` wants to replace the Debian numpy and **cannot**
  (`Cannot uninstall numpy 1.26.4, RECORD file not found`). Use apt.
* `cupy-cuda12x` installs but dies on `libcublas.so.12` — the box is CUDA
  13. **`cupy-cuda13x`** is the one.
* cupy's wheels are built against numpy 2; the system numpy is 1.26. A
  **venv** keeps that away from the running jobs, which use the system
  python. `python3 -m venv` fails until `python3.12-venv` is installed
  (`ensurepip is not available`).
* `pkill -f <pattern>` where the pattern occurs in your own ssh command line
  kills the shell (exit 255). Collect PIDs, kill by number.
* Killing a `sim.py` that a wrapper script `wait`s on makes the wrapper
  write its **completion stamp**. The pipeline then skipped the shape. Check
  for the artefact, not for the stamp.

## 3. The data: which corpus is which

**WikiText-2 raw, its own train / test split** (`bert_base/wikitext.py`,
from `Salesforce/wikitext` parquet). Calibrate on `train`, hold out `test`:
the rate is then about unseen prompts rather than about a change of domain,
and every shape is comparable. The earlier held-out sets were a different
Gutenberg book, which measures both at once.

`bert_base/vessl_data.sh` writes, for each T in 128 / 256 / 512:

| directory | what | drawn from | how |
|---|---|---|---|
| `all<T>/` | the weights + **1000 calibration prompts** | train | disjoint windows, seed 1 |
| `held<T>/serve/prompts/` | the **B = 65536/T prompts the card serves** | test | disjoint windows, seed 7 |
| `scan<T>/prompts/` | the **50,000-prompt failure scan** | test | RANDOM starts, seed 11 |
| `held_ref<T>/` | `calib.json` + `h_L{k}.f64` + `cls_logits.f64` | — | `sim.py`, `reference.py` |

Every prompt is `[CLS] body [SEP]` padded to T with a real length drawn
uniformly in `[3, T]`, so the mask and the `live` count are exercised.

Two things worth knowing about the draw:

* **The test split cannot give 50,000 disjoint windows.** It is ~280k
  wordpiece, which is ~2,200 windows at T = 128 and ~550 at T = 512. The
  scan therefore takes RANDOM starts (`--overlap`): 50,000 distinct prompts
  that OVERLAP, so the draws are correlated and the rate is not the rate of
  50,000 independent prompts. Say so when quoting it.
* **A big scan ships ids, not embeddings.** `[50000, 512, 768]` in f32 is
  **78 GB**; the ids are 100 MB and `Model.embed_ids` is a gather and a
  LayerNorm, which is the client's half anyway (`export.py --ids-only`).

**The calibration population is sized in PROMPTS, not tokens.** An earlier
decision here cut it to a constant token count (1000 at T = 128 → 250 at
T = 512) on the grounds that a window is a maximum over token positions.
That is wrong for the `shift` table, which is a maximum per (head, QUERY
POSITION): the number of entries grows with T, so a constant token count
gives each entry **four times fewer samples** at T = 512. Keep 1000 (500 is
the compromise at T = 512, where a calibration costs quadratically more
through the scores).

## 4. The loop: scan on the host before the card

`sim.py` WRITES a calibration. **`failure.py` READS one** and asks the
question a deployment asks: of N prompts it has never seen, how many does
this model serve? Every polynomial is rebuilt at the interval and degree
that ship, and every escape is attributed to the PROMPT it came from
(`Poly.escapes` is a global counter and cannot do this).

```
# the scan, on the card, 50,000 prompts in ~8 minutes
CUPY_GPU_MEMORY_LIMIT=8589934592 /root/venv/bin/python failure.py \
    /root/bert_base/all<T> /root/bert_base/held_ref<T>/calib.json \
    --ids /root/bert_base/scan<T>/prompts/ids.u32 --chunk 128 --f32 --gpu \
    --out .../gpu.json
# and the report
/root/venv/bin/python failure.py --merge '.../gpu.json' <T>
```

`--gpu` moves the whole chain onto the card by rebinding `model.np` and
`model._erf` to cupy (and `sim.np` with them — `ChoSoftmax` and `live_of`
live there). The weights load on the host because `np.fromfile` has no cupy
equivalent, then transfer. **It is the exact scan only**; `--approx` runs
the shipped polynomials too and is a subsample job on the CPU.

Three things the tool has to get right, and did not at first:

1. **A window's two ends fail differently.** Above the interval a Chebyshev
   polynomial is astronomical; below it a 1/sqrt SATURATES and returns a
   wrong answer in silence. Reporting a symmetric `max |t|` is actively
   misleading: a 1/sqrt window widened 5x at the bottom puts EVERY prompt at
   `t = -0.999` by construction, so the median prompt reads as one part in
   1e4 from escaping when it is nowhere near. Report the ends apart.
2. **One chunk goes through the whole chain**, not one layer through every
   prompt — the latter holds `[50000, 512, 768]` = 157 GB.
3. **Parts resume.** A long scan has to survive the overlay.

**Why an escape is the criterion.** Inside its interval a fit is good to its
`fit_err` (1e-5 or better here), far under the crypto's own noise, and
cannot decide anything. Outside, a degree-511 polynomial is astronomical.
So a prompt succeeds exactly when every one of its slots lands inside every
window — and the number beside the rate is the MARGIN, because that is what
says whether the exact-stream scan is conclusive (a slot at 0.7 cannot be
moved across by the chain's own 2^-10 perturbation; a slot at 0.999 might).

**`T * B = 65536`, so a per-prompt rate `p` is a per-BATCH rate
`1 - (1 - p)^B`.** One escaped slot is a batch-wide event. At T = 128 a
per-prompt 0.0060 % is a per-batch **3.03 %** at B = 512.

## 5. What the scan found, and the two knobs that answer it

Both are in `sim.py`; neither is a guess, and each is named by the scan.

**`--inv0-margin` — the first Cho window is the only one that escapes.**
At T = 256, all 105 escaped slots of 45,056 prompts are `inv0`; none in
`inv1`, `inv2`, the exp, the GELU or either LayerNorm. The fold's `est` is
a per-(head, token) geometric mean of the CALIBRATION population, so a
held-out row whose spread beats every calibration row's lands above the top
— measured at **10.9x the population maximum**. It is always the TOP, so
only that end is widened and the ratio grows by `im / margin` rather than
by its square.

**`--ln-margin` — a LayerNorm window is not a softmax window.** A held-out
batch whose every slot was INSIDE its windows (scan: worst top 0.956) still
blew up at layer 10 with 2^+130, because a 1/sqrt window's WIDTH is what
multiplies the input's error on the way out and `--margin 5.0` had layer 9's
`ln2` at **100,934x** against a raw 4,037x. The scan says the room is never
used: at every layer the largest variance maps to `t = -0.58`, i.e. the
held-out maximum is **1.04x the calibration's own** — a 4 % excess bought
with a 25x window. At `--ln-margin 1.5` the same layer's `ln1` drops from
degree 63 to **15** and `ln2` from 127 to **31**, which is levels and time
as well as error.

The shipping set is therefore

```
--margin 5.0 --ln-margin 1.5 --inv0-margin 30 \
--gelu-margin 1.8 --exp-margin 2.0 --exp-margin-hi 2.0 --inv-max-degree 255
```

`--inv-max-degree 255` is not taste: the LAST Cho pass's degree caps the
level plan (`P` lands at `top - 3 - Levels(deg)` and needs 4).

## 6. What the margins cost on the card

Measured at T = 128, B = 512, twelve layers, `ci16_35_k16_w58`:

| | oracle (one prompt, 512 instances) | held-out population |
|---|---|---|
| softmax | 5.4 s | **75–159 s** |
| LayerNorm | 2.0 s | **34–67 s** |
| bootstraps a layer | 1,536 | **3,072–6,144** |
| a layer | 104 s | **220–350 s** |

The whole difference is window width becoming polynomial degree: the wide
population windows put both inverse square roots at degree 255 and the
LayerNorms at 127–511, where the oracle's fold left the first Cho window at
a ratio of 1.7 and a low degree. **A window, its degree and its wall-clock
are one decision** — which is the other reason `--ln-margin` is worth
having.

## 7. The tests, and the exact experiment each one is

Five programs, one of which is the only thing that touches ciphertexts.

| what | is | answers |
|---|---|---|
| `bert_base/export.py`, `wikitext.py` | host | the corpus, the weights, the prompt sets |
| `bert_base/reference.py` | host, float64 | `h_L{k}.f64` — what the encrypted run is SCORED against |
| `bert_base/sim.py` | host | **writes** `calib.json`: every window, degree, shift, fold and ride |
| `bert_base/failure.py` | host | **reads** one: how many held-out prompts does it serve? |
| `unittest/CiBertBaseTest.cpp` → **`ci_bert_base_test`** | the A100, real CKKS | the encrypted twelve layers + head against `h_L{k}.f64` |

The crypto test is one gtest, `CiBertBase.TheChainRunsOnTheRealWeights`,
and it is driven entirely by environment (`bert_base/vessl_held.sh` is this,
repathed). Nothing about the shape is compiled in: `B` follows from the
slot count and `T`, `N` from the input file's size.

```
CHEDDAR_POOL_MIB=0 CHEDDAR_POOL_RELEASE_MIB=0 CHEDDAR_POOL_MAX_BIN_MIB=0 \
BERT_BASE_ALL=/root/bert_base/all<T>            # weights + meta.json
BERT_BASE_REF=/root/bert_base/held_ref<T>       # h_L{k}.f64, cls_logits.f64
BERT_BASE_CALIB=/root/bert_base/held_ref<T>/calib.json
BERT_BASE_INPUTS=/root/bert_base/held<T>/serve/prompts/inputs.f32
BERT_BASE_MASK=/root/bert_base/held<T>/serve/prompts/mask.u8
BERT_BASE_LAYERS=12 BERT_BASE_HEAD=1 ./ci_bert_base_test
```

Other knobs it reads, none of which this session needed to move:
`BERT_BASE_PARAM` (the ring, default `ci16_35_k16_w58`), `_FIRST_LAYER`
(start the chain at layer L from `h_L{L-1}.f64` — how a single layer is
isolated), `_INSTANCES`, `_BOOT_GROUP`, `_POLY_BATCH`, `_FOLD`, `_HOIST`,
`_DUMP` / `_DUMP_INSTANCES` (tapped intermediates for `debug.py`),
`_PRINT_LABELS`, `_LABELS_OUT`.

**The five experiments, in the order they have to happen.**

1. **Bring-up** — `vessl_setup.sh <branch>`: apt, cmake 3.31.6, clone,
   configure, build the one target.
2. **Data** — `vessl_data.sh`: OpenBLAS, scipy, pyarrow, WikiText-2, and
   three `export.py` runs a shape (calibration population, serve set, scan
   ids).
3. **Calibration + reference** — `sim.py --no-chain` writes `calib.json`;
   `reference.py` writes `h_L{k}.f64` and `cls_logits.f64` for the serve
   set. `vessl_host.sh <T>` runs both at once.
4. **The scan** — `failure.py ... --gpu`, 50,000 held-out prompts, then
   `--merge` for the rate. **This decides whether step 5 is worth a GPU
   hour**, and this session it twice said no.
5. **The encrypted run** — `vessl_held.sh <T>`.

The validation of anything that moved onto the card was the same in both
cases: **run it both ways and diff.**

* `sim.py --gpu` against `sim.py` on the same 64 prompts: every degree,
  every channel permutation and every suppression **identical**, and the
  worst relative difference in any window bound **6.0e-15**.
* `failure.py --gpu` against `failure.py` on the same 256 prompts: **0 / 256
  escaped, worst top 0.841** from both.

## 8. What everything cost, measured

A100 + 11 cores, this session, `ci16_35_k16_w58`. "projected" means the rate
was measured over one or more layers and the rest extrapolated — every other
number is a wall clock.

**Setup and data, once per empty overlay**

| step | time |
|---|---|
| `vessl_setup.sh` (apt + clone + configure 22 s + build `-j48`) | **2 min 33 s** |
| `vessl_data.sh` (WikiText-2 + 3 exports x 3 shapes) | **~17 min** |
| `reference.py`, 12 layers + head, the serve set | ~25 min (host, concurrent) |

**Calibration — `sim.py --no-chain`, the whole population, 12 layers**

| T | prompts | on 11 cores | **on the A100** |
|---|---|---|---|
| 128 | 1000 | 76 min (6.3 min a layer) | **19 s** |
| 256 | 1000 | 3.8 h projected (19 min a layer) | **32 s** |
| 512 | 500 | 10 h projected (>50 min a layer) | **42 s** |
| 512 | 300 | — | **31 s** |

The CPU column is why `sim.py` grew a `--gpu`: a calibration is quadratic in
T through the `[N, NH, T, T]` scores, and this box has eleven cores.

**The scan — `failure.py`, 50,000 WikiText-2 test prompts, 12 layers**

| T | chunk | on the A100 | on 11 cores |
|---|---|---|---|
| 128 | 128 | **474 s / 500 s** (two runs) | 64 min projected |
| 256 | 64 | **~19 min** (40,960 + a 188 s resume) | ~2 h projected |
| 512 | 32 | **2,420 s = 40 min** | ~4.5 h projected |

Host per-prompt cost at T = 128, measured single-threaded: **1.59 core-s
before scipy, 0.84 after**. A 4,096-prompt GPU probe ran in **17 s**.

**The encrypted runs — `ci_bert_base_test`, 12 layers + head, held out**

| shape | setup | a layer | head | **total** | result |
|---|---|---|---|---|---|
| (512, 128) | 22.0 s | 175–509 s | 53.3 s | **76 min 28 s** | PASSED, labels 512/512 |
| (256, 256) | 7.2 s | 224–552 s | 53.3 s | **76 min 10 s** | PASSED, labels 255/256 |
| (128, 512) | 7.4 s | 375–961 s | 53.0 s | **114 min 32 s** | PASSED, labels 128/128 |

A layer's spread is the softmax's Cho pass count `k`, which the calibration
picks per layer and which sets the bootstrap count almost alone:

| T | `k` per layer | the expensive ones |
|---|---|---|
| 128 | `223222223222` | L2, L8 at k = 3 |
| 256 | `234222233232` | **L2 at k = 4** (552 s, 11,520 boots), five at k = 3 |
| 512 | `224332222222` | **L2 at k = 4** (961 s, **20,736 boots**), L3/L4 at k = 3 |

Bootstraps a layer ran 2,304 (k = 2, T = 128) to 20,736 (k = 4, T = 512),
and boot is 60–88 % of a layer throughout. **The run that FAILED** — the
same (512, 128) on the margins that shipped before — took 84 min to reach
2^+130 at layer 10, which is the whole argument for step 4 costing eight
minutes first.

**The speed work, and what each piece bought**

| | before | after |
|---|---|---|
| numpy's BLAS (reference → OpenBLAS) | 2 GFLOPS | **140 GFLOPS** |
| the GELU's `erf` (`np.vectorize` → scipy) | 5.34 s | **0.112 s** (48x) |
| one prompt's 12-layer host chain | 1.59 core-s | **0.84 core-s** |
| a T = 512 calibration | 10 h | **42 s** |
| a 50,000-prompt scan at T = 128 | 64 min | **8 min** |

## 9. Running it, from an empty overlay

```
scp bert_base/vessl_setup.sh vessl:/root/ && ssh vessl \
  'setsid bash -c "bash /root/vessl_setup.sh BERT_base" < /dev/null &'   # 2.5 min
ssh vessl 'cd /root/work/cheddar-bb && bash bert_base/vessl_data.sh'     # ~17 min
ssh vessl 'cd /root/work/cheddar-bb && bash bert_base/vessl_host.sh 128' # f64 reference
# the calibration itself, on the card (seconds, not hours):
ssh vessl 'cd /root/work/cheddar-bb/bert_base && /root/venv/bin/python sim.py \
   /root/bert_base/all128 /root/bert_base/held_ref128/calib.json \
   --inputs /root/bert_base/all128/prompts/inputs.f32 --no-chain --gpu \
   --margin 5.0 --ln-margin 1.5 --inv0-margin 30 --gelu-margin 1.8 \
   --exp-margin 2.0 --exp-margin-hi 2.0 --inv-max-degree 255'
# the scan, and ONLY if it is clean:                                     # 8 min
ssh vessl 'cd /root/work/cheddar-bb && bash bert_base/vessl_held.sh 128'  # 76 min
bash bert_base/pull.sh          # calibrations and scan parts to the laptop
```

`vessl_pipeline.sh` chains a shape's scan and run, one card job at a time.
Strip CR after every `scp` of a `.sh` (`sed -i 's/\r$//'`) — a file edited
on Windows dies on line 1. And `CUPY_GPU_MEMORY_LIMIT` matters when a scan
runs beside an encrypted run: the run holds 40–46 GB of the 80, the scan
needs 4 GB at T = 128 and ~20 GB at T = 512, and a calibration at T = 512
wants 25–32 GB for its score buffer alone (`--limit` and `--chunk` are the
knobs when it will not fit).
