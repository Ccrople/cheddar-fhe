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

**2026-09-11, the three service shapes.** Same binary, same ring, same
calibration recipe (`sim.py` on the batch's own prompts, k = 2): only
`meta.json`'s T changes the layout, the rotation stride and the BSGS split.

| (T, B) | prompts | layer 0 | layer 1 (chained) | wall / layer | wide boots / layer | softmax |
|---|---|---|---|---|---|---|
| (128, 512) | 512 held-out | 2^-10.07 (worst 2^-9.75) | 2^-9.37 (worst 2^-7.74) | 26 / 31 s | 384 / 512 | 11.5 s |
| (256, 256) | 256 (own) | 2^-9.53 (worst 2^-9.05) | 2^-9.02 (worst 2^-7.66) | 39 / 44 s | 640 / 768 | 22.9 s |
| (512, 128) | 128 (own) | 2^-8.81 (worst 2^-8.34) | 2^-8.41 (worst 2^-7.03), k = 3 | 64 / 112 s | 1152 / 2304 | 45 / 89 s |

Everything but the softmax is flat in T (GELU 4.7 s, projections 0.4 s,
LN 0.5 s at every shape; the diagonal products 1.6 -> 3.5 s); what grows
is the softmax's main-path boots, `2T(k-1)` a head-pair, and the sim
raised k to 3 at T = 512's layer 1 (its k = 2 first window exceeded the
300x rule). Per token-layer: 0.4 ms (T 128), 0.7 ms (256), 1.0 / 1.7 ms
(512). The crypto floor also slides with T (2^-10 -> 2^-8.4) while the
host approximation-only chain stays at 2^-17..-21: it is boot noise on the
main path through the softmax's and norms' Jacobians. The levers, in
order: (1) the `cho_est`-style per-row fold of the first window, which is
what took T = 4096's k down and which brings k back to 1-2 here; (2) fewer
main-path boots by landing the last pass without one; (3) hoisted
rotations and `EvaluateBatch` for the GELU (4.7 s of every layer).
Rotations / relinearizations a layer: 6080 / 2566 (T 128), 8640 / 4870
(256), 13696 / 7430-9480 (512).

Where the host work runs: `sim.py` on 1000 + 1000 prompts took 10 min on
the laptop (numpy, one core); the T = 256 / 512 calibrations ran on vessl
(96 cores) -- do that by default.

Probe table of the first correct run (layer 0, recorded prompt, rms bits):
q 13.7, k 14.3, v 13.5, scores 13.0, exp 13.5, sq 14.1, r 13.2, P 11.7,
attention out 12.6, O 12.3, residual 13.1, LN1 var 14.6, r 15.5, LN1 out
10.5 (the empty-instance leak), GELU input 11.1.

## 5b. The head, the padding mask, serve mode (2026-09-11, later)

* **Head** (`CiBertTinyLayer::Head`): the last layer's output booted, the
  pooler as a Kang GEMM (128 -> 128) with the tanh's affine folded, tanh as
  ONE polynomial on the pooler input's CERTIFIED interval, the classifier
  as a Kang GEMM (128 -> 2) with its bias; instance b's logits sit at its
  [CLS] slot (token 0). The interval is a theorem of the last LayerNorm:
  `z = g n + b` with `n` centred and `|n| <= sqrt(H)`, so `|u_j - c_j| <=
  sqrt(H) ||g W_j - mean||` for every prompt there is -- BERT-Tiny: [-26.2,
  24.9], where the recorded prompt reaches 5.07 -- and the polynomial there
  is degree 255 (8 levels), which the ladder affords (16 -> 15 -> 7 -> 6).
  Every other token's slot is the same head on that token, also inside the
  interval, so nothing can escape.
* **Padding mask** (`SetMask`): pads are projected, scored and exp'd like
  every key (the calibration's exp domain covers them: `sim.py` takes the
  domain over ALL keys and the shift over the REAL ones), then `y_d` is
  multiplied by a 0/1 per-slot plaintext `M_d[t][b] = valid[b][(t+d) mod T]`
  before the row sum. One wide level, reserved by the plan. T plaintexts,
  rebuilt only when the level changes. Padded positions are still computed
  (as BERT does) and never read.
* **Serve mode**: `bert_tiny/serve.sh <text.txt>` = `encode.py` (the
  client: WordPiece from the exported vocab, embeddings, embedding LN,
  padding to T, `inputs.f32` + `mask.u8`) -> the run with the population
  calibration -> logits and labels per line; the float64 reference runs
  beside it so every logit is checked. The head's label is BERT-Tiny's
  pretrained NSP classifier (is-next-sentence), the only classifier the
  checkpoint ships; a fine-tuned task head is the same two GEMMs with its
  own weights.

Ledger additions: mask = 1 wide level (choice: multiplicative on `y`);
tanh interval = certified (theorem), degree by tol; the calibration is now
the PADDED population (1000 prompts with random real lengths).

**Measured (A100, T = 128, B = 512, 2 layers + head):**

| run | calibration | layers | head (7.7 s) |
|---|---|---|---|
| recorded prompt x 512 | oracle | 2^-10.22 / 2^-9.43 | logits 2^-9.92, labels 512 / 512 |
| 512 held-out, padded 16..128, masked | 1000 padded prompts (k = 3) | 2^-9.89 / 2^-9.37 (worst 2^-8.01) | logits 2^-7.25, labels 512 / 512 |
| `demo.txt` (8 lines, 9..56 real tokens) | the same | **2^+236** | garbage |

The demo's failure is the calibration rule again, from the other side: its
shortest line has 9 real tokens, the calibration's prompts had at least
16, so the first Cho window (a statistic of the row sum, which shrinks
with the number of real keys) was escaped by ONE prompt, the degree-63
polynomial there is ~1e26, and every instance of the batch came back as
garbage. The run now records the calibration's shortest prompt
(`min_real_tokens`) and WARNS when a served prompt is shorter; the
calibration is re-cut with lengths 3..128 (section 5c).

## 5c. The calibration re-cut with lengths 3..128, and the demo

`export.py --min-len 3` (1000 prompts of *War and Peace*, 1000 held out of
*Great Expectations*, real lengths uniform in 3..128). The sim now picks
Cho k = 3 (L0) and k = 4 (L1): with three real keys the row sum's window is
what it is, and the fold that would narrow it (`cho_est`) is the next
lever. Host held-out: 2^-18.7 / -17.0, head labels 1000 / 1000.

| run | layers | head |
|---|---|---|
| 512 held-out, lengths 3..128, masked (k = 3 / 4) | 2^-9.90 / 2^-9.00 (worst 2^-8.23), 36 + 52 s | logits 2^-7.53, labels 512 / 512 |
| `demo.txt`, 8 lines of 9..56 real tokens, `serve.sh` | 2^-10.34 / 2^-9.46, 36 + 52 s | logits 2^-9.04, labels 8 / 8 (512 / 512 instances) |

`serve.sh demo.txt` end to end: encode (client) 1 s, keys + tables 8 s,
two layers 88 s, head 8 s, decrypt; the labels file lists each line's two
logits and its label (line 3, "call me ishmael...", is the one the NSP
head calls class 1). Every logit is within 2^-9 of the float64 model's.

## 5d. The four levers (2026-09-12)

All four are switches, all four are A/B-able from the environment, and the
accuracy is the same on both sides of every one of them:
`BERT_TINY_FOLD` (default 1, inert unless the calibration carries an
estimate), `BERT_TINY_HOIST` (default 1), `BERT_TINY_POLY_BATCH` (default
1; 8 is the measured knee), and the ring and its secret are
`BERT_TINY_PARAM`.

### The Cho fold

`1 / sqrt(sq) = (1 / sqrt(est)) (1 / sqrt(sq / est))` for any public
`est > 0`, so the inverse square root can be given the RATIO and its window
becomes what ONE ROW's population spread is instead of the whole
population's. The estimate is

    est[b][t] = ehat[head][t] * live_b ^ p ,   p = 1 at pass 0, else 0

with `live_b` the instance's real-token count -- which the padding mask
makes public -- and `ehat` the calibration's per-row geometric mean of
`sq / live` (that choice is the one that MINIMISES the ratio window: it
leaves `[sqrt(lo/hi), sqrt(hi/lo)]` per row). The length dependence of
`sq_0 = a sum over exactly live_b keys` therefore leaves the window
EXACTLY, which is the thing a short served prompt used to escape (5c).

It costs no wide level and no rotation: `1/est` rides the scaling that
already precedes the narrow path's bootstrap, and `1/sqrt(est)` rides the
constant already on `r` -- both on the ONE ciphertext a head's row sums
live in. Only the LAST pass pays a narrow level it did not pay before,
which is why `sim.py --fold-passes` defaults to `first`; the later windows
are near-theorems and the fold buys ~1.2x there.

What it does to the windows (1000 padded prompts, lengths 3..128):

| | pass 0 raw | pass 0 folded | pass 1 raw | pass 1 folded |
|---|---|---|---|---|
| layer 0, k = 1 | 20,100 x | **340 x** | | |
| layer 0, k = 2 | 901 x | **14.6 x** | 91.4 x | 75.2 x |
| layer 1, k = 1 | 47,700 x | **659 x** | | |
| layer 1, k = 2 | 2,100 x | **50 x** | 107 x | 51.3 x |

The auto-`k` rule is "the first window under 300x", so without the fold
layer 1 needs k = 3 and with it k = 2 -- and one Cho pass is 2T = 256
main-path bootstraps a layer.

### Hoisted BSGS rotations, and the batched polynomials

The diagonal products rotate ONE ciphertext by several distances: the baby
set `rot(K_i, b B)` for b < baby, the giant set `rot(Q_i, -g baby B)` for
g < giant. `CiBertTinyLayer::RotateMany` decomposes such a source ONCE
(`ModSwitchHandler::ModUp` plus the b-part's `PseudoModUp`) and then pays
only the key product, the mod-down and the permutation per distance -- word
for word what `Context::MultKey` does, since `MultKeyNoModDown` takes the
decomposition as an argument. Of the 6080 rotations a layer at T = 128,
4736 run off 384 decompositions.

`EvalMany` is `EvalPoly::EvaluateBatch` over a group of ciphertexts at one
(level, scale): the exp over the T diagonals, the GELU over the 512 hidden
channels, the tanh over the head's 128.

Measured (A100, `ci16_35_k16_w58`, T = 128 x B = 512, the oracle
calibration, 2 layers, seconds):

| | L0 rms | L1 rms | L0 | L1 | scores | values | GELU L0 |
|---|---|---|---|---|---|---|---|
| neither (the 5a baseline) | 2^-10.22 | 2^-9.33 | 15.26 | 18.95 | 1.62 | 1.10 | 4.70 |
| hoist | 2^-10.22 | 2^-9.31 | 14.76 | 18.48 | **1.33** | **0.98** | 4.67 |
| hoist + batch 8 | 2^-10.22 | 2^-9.36 | **13.51** | **17.68** | 1.34 | 0.99 | **3.37** |
| hoist + batch 32 | 2^-10.22 | 2^-9.36 | 13.58 | 17.73 | 1.33 | 0.98 | 3.26 |

Hoisting takes the diagonal products down 18 % / 11 %; the batch takes the
GELU down 28 % and stops paying past 8 (past 32 it costs boot memory). The
layer is 11.5 % faster and every rms is the same to the run-to-run draw.
(The batched Llama layer's own note records `EvaluateBatch` as FLAT for its
degree-15 exp; it pays here because BERT-Tiny's GELU is degree 127 over 512
ciphertexts, where the tree walk and not the card is the cost.)

### What the secret is worth, in bits

`reference/scripts/security_estimate.py` on this ring (primal uSVP plus the
drop-columns sparse hybrid; an UPPER bound on security, since the published
MITM hybrids are stronger):

| secret | where it is published | classical | quantum |
|---|---|---|---|
| MAIN, h = 32768 | ciphertexts (logQ 1400) | 166.0 | 154.1 |
| MAIN, h = 32768 | switching keys (logQP 1771) | 128.4 | 119.9 |
| SPARSE, h = 32 (the preset) | the same logQP | **93.4** | -- |

The binding number for BERT-Tiny as shipped is therefore **93.4 bits**, and
it is the SPARSE secret the bootstrap's encapsulation publishes, not the
main one. Raising it is not free, because the sparse weight and EvalMod's
range K are ONE decision -- measured on the same 2-layer chain:

| ring | sparse h | security | layer 0 | layer 1 | wall |
|---|---|---|---|---|---|
| `ci16_35_k16_w58` (K 16) | 32 | 93.4 | 2^-10.22 | 2^-9.33 | 14.8 / 18.5 s |
| `ci16_35_k16_w58_h64` (K 16) | 64 | 121.1 | **2^+235.7** | 2^+270.3 | -- |
| `ci16_35_k64_w58_h192` (K 64) | 192 | **128.4** | 2^-10.63 | 2^-9.58 | 14.6 / 19.4 s |

The middle row is the mechanism made visible: at h = 64 the ModRaise
wrap-around leaves K = 16 and the bootstrap returns garbage for every
instance of the batch. On the K = 64 pool, h = 192 is enough that the
hybrid buys nothing (drop 0), so the answer is **128.4 bits at +2 % wall
and no accuracy loss** -- the k64 pool's boot is ~10 % dearer and its
landing is 14 rather than 16, which this layer's plan has room for.

## 6. Plan

1. B = 1, T = 128 on the A100: build, layer 0, the 2-layer chain. DONE.
2. Population calibration on 1000 prompts, 512 held-out on the crypto.
   DONE (k = 2, 2^-10.07 / -9.37).
3. The three service shapes. DONE (table above); T = 512 wants the
   first-window fold to keep k at 2.
4. The head, the padding mask and serve mode. DONE (5b, 5c): text in,
   labels out, every logit checked.
5. Next, in order: the `cho_est` fold (the first Cho window is the one
   statistic left that a short prompt can escape; folding the per-row
   estimate out takes k from 3-4 back to 2 and the softmax's boots with
   it); per-channel certified GELU intervals (the sphere bound: nothing
   per prompt, no escape possible); hoisted rotations and the batched
   GELU (4.7 s of every layer); a fine-tuned task head (the same two GEMMs
   with its weights); the security statement for the sampled secret.
