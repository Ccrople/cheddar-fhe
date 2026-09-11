# BERT-Tiny on the batched layout (branch `BERT_tiny`)

The plan, what is built, what was measured, and -- in one place -- every
number that is not a weight: the **knob ledger** (section 4). Everything
below the level of "which algorithm" lives there, so nothing is tuned that
is not written down.

## 1. The shape and the three targets

BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`): 2 layers, H = 128, 2 heads
of 64, FFN 512, post-LN, GELU (erf), LayerNorm eps 1e-12, positions up to
512. The three service shapes are (T, B) = (128, 512), (256, 256),
(512, 128) -- all `T * B = 65536 = N`, the slot count of `ci16_35_k16_w58`
(R+, logN 16, scale 2^35, boot landing 16). **B = 1, T = 128 is the same
code** with one live instance and 511 empty ones: the layer's work does not
depend on how many are live, so the B = 1 numbers ARE the B = 512 numbers.

## 2. The design (one layout, one ring, no ring switch)

Layout `[channel][token][batch]`: ciphertext = channel, slot
`s = t * B + b` (instance fastest; `CiBatchLayout`, plain map). Three
things follow and are the whole reason for it:

* a token shift by `d` inside EVERY prompt is ONE cyclic slot rotation by
  `d * B` (T * B = N closes the wrap inside the prompt, no masks);
* a per-token constant (the softmax's row shift) is one plaintext, the
  same at every instance;
* every channel reduction (LayerNorm's mean and variance, the softmax's
  row sum) is a SUM OF CIPHERTEXTS: no rotation, no key, no level.

| operator | algorithm | memory | levels (wide) |
|---|---|---|---|
| Q/K/V, O, FFN up/down | [KANG] Alg. 1 (`CiBatchProjection`: int8 GEMM over whole ciphertexts, no key); biases are constants | `[in ch]` -> `[out ch]` cts | 1 each |
| Q K^T | **diagonal product** `S_d = sum_c Q_c (.) rot_{dB}(K_c)`, d = 0..T-1, BSGS over d (`baby x giant = T`); T relins a head | `[head][d][query t][b]`: T cts a head | 1 |
| softmax | Cho: `y = exp((S - shift)/2^k)`, k passes of `y <- (y/|y|_2)^2`; shift = public per-(head, query t) plaintext; sum over keys = T ct adds; 1/sqrt on ONE ct (narrow path, booted when it needs levels) | T cts a head | exp + 2 a pass (+ a wide boot between passes) |
| P V | the same product transposed: `O_c = sum_d P_d (.) rot_{dB}(V_c)` | `[channel][query t][b]` = the layout | 1 |
| LayerNorm | `H x_c - sum` (integer multiply, free), variance = one relinearized sum of squares, 1/sqrt on that ONE ct (any degree), gain rides r's 128 scalar copies, bias a constant | 128 cts | 1 |
| GELU | one Chebyshev polynomial per hidden ct (today one shared interval; per-channel certified intervals = the same code with 512 fits) | 512 cts | ceil(log2(deg+1)) |

Units: every ciphertext carries a public `carry` (message = carry x model
value). A bootstrap input is scaled so that |message| <= `ride` (0.35): the
stream entering a layer carries `ride / in_absmax`, the LN output
`ride / out_absmax`, the softmax's main path `ride` through `sqrt(ride)` on
`r`, and the narrow accumulators `ride / their maximum`. Projections fold
`carry_out / carry_in` into the weight; the softmax folds its exp affine
into W_Q and the shift; the norms fold `1/(H^3 carry^2)` into the variance
affine. Nothing else about units exists.

**Level ledger** (landing 16, k = 1, exp deg 15 = 4 lv): x booted -> 16;
q/k at 11, S at 10; y at 6; sq 5 -> narrow boot -> r 9; w = y r 5; P 4;
P V 3; O 2; residual 2; LN1 -> 1 (one wide level); boot -> 16; up 15;
GELU(127, 7 lv) 8; down 7; residual 7; LN2 -> 6. With k >= 2 the main
path is booted between passes, so S needs only exp + 2.

Boots a layer: wide `128 + 128 + 2T(k-1)`, narrow `2k + 2`. At k = 1 that
is 256 wide + 4 narrow, and the softmax has NO wide boot.

## 3. What is built

* `bert_tiny/export.py` -- checkpoint -> f32 blobs; the shape from the
  checkpoint's `config.json`; `--prompts N --corpus text` writes a
  population's inputs.
* `bert_tiny/model.py` -- the float64 forward, batched over prompts, with
  hooks; `reference.py` writes `h_L{k}.f64` (+ raw stats).
* `bert_tiny/sim.py` -- the chain on the host with the approximations in:
  picks every degree (smallest of 7..255 under `tol`) and window (with
  margins), reports the chain's rms per layer, writes `calib.json`. This
  is where every knob is DECIDED; the C++ only reads it.
* `extension/CiBertTiny.*` -- `CiBertTinyLayer` (the layer above);
  `unittest/CiBertTinyTest.cpp` -- the chain against `h_L{k}.f64`
  (`BERT_TINY_ALL`, `BERT_TINY_REF`, `BERT_TINY_LAYERS`,
  `BERT_TINY_INSTANCES=1|all`, `BERT_TINY_PARAM`, `BERT_TINY_INPUT_LEVEL`).

## 4. The knob ledger

Everything that is not a weight or an algorithm. Column "who decides"
says whether the number is a THEOREM (fixed by the algebra), a CALIBRATION
(from data, currently the recorded prompt = oracle; item 4 of the plan
replaces it with 1000 prompts), a BUDGET (a level or ride bound) or a
CHOICE (mine, revisable).

| knob | value (B=1, T=128, the recorded prompt) | who decides | where |
|---|---|---|---|
| ring | `ci16_35_k16_w58` (landing 16, K 16, 4 CtS) | choice | test `BERT_TINY_PARAM` |
| ride (boot input height) | 0.35 | budget (EvalMod cubic 2^-15) | `Config::ride` |
| boot group | 8 | choice (BootBatch exact) | `Config::boot_group` |
| BSGS split | 16 x 8 (T=128), 16 x 16, 32 x 16 | choice | `Config::baby_steps` |
| Cho passes k | 1 (L0, L1) | calibration: smallest k with first window under `sq_ratio` 300 | sim `--k`, `--sq-ratio` |
| row shift | per (head, query t) max over prompts and keys | calibration | calib `softmax.shift` |
| exp domain | L0 [-6.11, 0.25], L1 [-13.51, 0.25] (= range of (S - shift)/2^k, lo - 1.0, hi + 0.25) | calibration + margin | sim `--exp-margin` |
| exp degree | 15 (both layers) | tol 1e-5 abs | sim `--tol` |
| 1/sqrt window, pass 0 | L0 [1.42, 56.2], L1 [0.77, 26.2] (range x 1.3 each side) | calibration + margin | sim `--margin` |
| 1/sqrt degree, pass 0 | 63 (L0), 31 (L1) | tol 1e-5 rel | sim |
| LN1 var window / degree | L0 [1.13, 3.45] / 15, L1 [1.05, 5.25] / 15 | calibration + margin / tol | sim |
| LN2 var window / degree | L0 [1.32, 9.07] / 15, L1 [0.88, 2.15] / 7 | same | sim |
| GELU interval / degree | L0 +-14.7 / 127, L1 +-13.2 / 63 (max |u| x 1.2) | calibration + margin / tol | sim `--gelu-margin` |
| stream carries | in: ride / in_absmax; h: ride / ln1.out_absmax; z: ride / ln2.out_absmax | budget from calibration maxima | `Prepare` |
| narrow boot scales | sq: ride / window hi; V': ride / (var hi / kappa0); r: ride / r_max | budget | `NarrowLift` |
| level plan | S at exp + 6 (k=1) or exp + 2; P >= 4; GELU under top - 2 | budget (asserted) | `Prepare` |
| padding mask | none (every prompt exactly T real tokens) | choice | -- |
| secret | the library's default sampled by `UserInterface` | choice (security to be stated) | test |

Approximation-only chain (sim, no crypto noise): L0 2^-20.6, L1 2^-17.7.

## 5. Measured (A100, `ci16_35_k16_w58`, T = 128, B = 512)

**2026-09-11, the recorded prompt in every instance (= B = 1 done 512 times),
the oracle calibration of section 4:**

| | rms vs float64 | worst instance | wall | boots | of which |
|---|---|---|---|---|---|
| setup (keys, tables) | | | 7.7 s | | |
| layer 0 | **2^-10.22** | 2^-10.02 | 15.2 s | 128 wide + 4 narrow | boot 5.4, GELU 4.7, scores 1.6, values 1.1, softmax 0.9 |
| layer 1 (chained) | **2^-9.37** | 2^-7.63 | 19.0 s | 256 wide + 4 narrow | boot 10.7, GELU 3.3, scores 1.6, values 1.1, softmax 1.0 |

6080 rotations and 2052 relinearizations a layer. Per instance-layer that
is 30-37 ms; the layer's cost does not depend on how many instances are
live, so these are the B = 512 numbers too. The approximation-only chain
(sim) is 2^-20.6 / 2^-17.7: the crypto floor (boot noise through the
softmax's and the norms' Jacobians) is what the 2^-10 is.

**The rule found the hard way -- THE BATCH IS ALWAYS FULL.** With one live
instance and 511 empty ones the layer came back at 2^+45. Probes (section
3's `debug.py`) put every stage through the GELU's INPUT at 2^-11..-15 and
exactly one hidden channel of the GELU's output at 2^+18. An empty instance
is an all-zero prompt: its LayerNorm variance is 0, outside every window,
where the degree-15 Chebyshev is 1.5e8; its centred value is a noisy zero,
so its norm output is ~1e5 instead of the bias, and the GELU then runs off
its interval there. CKKS noise lives in the coefficient domain, so ONE bad
instance pollutes every slot of the ciphertext -- the live prompt included
(already visible at LN1: 2^-10.5 against 2^-13 elsewhere). Consequences:
(a) unused instances carry a real prompt (the default now, and the leak
check for free); (b) any window escape in any prompt of the batch is a
batch-wide event, which is the argument for certified intervals (the
GELU's sphere bound, Cho's theorem windows) over statistics.

**2026-09-11, 512 DISTINCT held-out prompts on the POPULATION calibration
(goal item 4).** Calibration: 1000 disjoint 126-piece windows of *War and
Peace* (`export.py --prompts 1000`), `sim.py` picking Cho k = 2 (the k = 1
first window exceeds the 300x rule), exp deg 15 on [-4.5, 0.25] / [-10.3,
0.25], 1/sqrt windows [0.92, 76.7] deg 63 + [0.0061, 0.149] deg 31 (L0) and
[0.14, 43.5] deg 127 + [0.0062, 0.90] deg 127 (L1), LN windows up to 16x
at deg 15/31, GELU +-16.3 deg 127. Held out: 1000 windows of *Great
Expectations* -- host chain 2^-21.2 / 2^-19.2 with 2 + 10 tiny escapes;
the first 512 of them on the A100, each instance against its own float64:

| | rms | worst instance | wall | boots |
|---|---|---|---|---|
| layer 0 | **2^-10.07** | 2^-9.75 | 26.0 s | 384 wide + 4 narrow |
| layer 1 (chained) | **2^-9.37** | 2^-7.74 | 30.7 s | 512 wide + 4 narrow |

Same accuracy as the oracle; the second Cho pass costs 2T = 256 wide
boots a layer (softmax 0.9 -> 11.5 s). The crypto floor, not the
calibration, is the 2^-10. What the population moved (the knob ledger's
"calibration" rows): k 1 -> 2, every window wider, LN2 at L0 deg 15 -> 31,
GELU L1 deg 63 -> 127.

Probe table of the first correct run (layer 0, recorded prompt, rms bits):
q 13.7, k 14.3, v 13.5, scores 13.0, exp 13.5, sq 14.1, r 13.2, P 11.7,
attention out 12.6, O 12.3, residual 13.1, LN1 var 14.6, r 15.5, LN1 out
10.5 (the empty-instance leak), GELU input 11.1.

## 6. Plan

1. B = 1, T = 128 on the A100: build, layer 0, the 2-layer chain. <- here
2. Population calibration: `export.py --prompts 1000 --corpus ...`,
   `sim.py --inputs ... --held-out ...`; the softmax's k rises with the
   population (k = 2 expected), the windows widen, `cho_est`-style folds if
   the first window is wide.
3. The three service shapes: `export.py --tokens 256/512` (positions up
   to 512 are in the checkpoint); only the rotation stride, the BSGS split
   and the calibration change.
4. Per-channel certified GELU intervals (the sphere bound), padding masks,
   the pooler/classifier head.
