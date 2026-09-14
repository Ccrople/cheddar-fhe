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
| `python3.12-venv` then a venv with `numpy` `scipy` **`cupy-cuda13x`** | the scan on the card | **50,000 prompts: 80 min → 8 min** |

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

## 7. Running it, from an empty overlay

```
scp bert_base/vessl_setup.sh vessl:/root/ && ssh vessl \
  'setsid bash -c "bash /root/vessl_setup.sh BERT_base" < /dev/null &'   # ~3 min
ssh vessl 'cd /root/work/cheddar-bb && bash bert_base/vessl_data.sh'     # ~15 min
ssh vessl 'cd /root/work/cheddar-bb && bash bert_base/vessl_host.sh 128' # calib + f64 ref
#   ... then the scan (section 4), and only then:
ssh vessl 'cd /root/work/cheddar-bb && bash bert_base/vessl_held.sh 128'
bash bert_base/pull.sh          # calibrations and scan parts back to the laptop
```

`vessl_pipeline.sh` chains the last three per shape, one card job at a time.
Strip CR after every `scp` of a `.sh` (`sed -i 's/\r$//'`) — a file edited
on Windows dies on line 1.
