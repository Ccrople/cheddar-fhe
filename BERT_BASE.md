# BERT-Base on the batched layout (branch `BERT_base`)

The plan, what is built, what was measured, and -- in one place -- every
number that is not a weight: the **knob ledger** (section 4). Everything
below the level of "which algorithm" lives there, so nothing is tuned that
is not written down.

This is `BERT_tiny`'s design at 6x the width and 6x the depth. Read
`BERT_TINY.md` first if you want the layout's argument; this file records
only what BERT-Base changes, and it changes four things.

## 1. The shape and the four targets

BERT-Base (`google-bert/bert-base-uncased`): **12 layers, H = 768, 12 heads
of 64, FFN 3072**, post-LN, GELU (erf), LayerNorm eps 1e-12, positions up to
512, plus the pretrained pooler and NSP classifier. The service shapes are
(T, B) = **(128, 512), (256, 256), (512, 128)** -- all `T * B = 65536 = N`,
the slot count of `ci16_35_k16_w58` (R+, logN 16, scale 2^35, boot landing
16) -- and **B = 1, T = 128 is the same code** with one prompt in every
instance: the layer's work does not depend on how many are live, so the
B = 1 numbers ARE the B = 512 numbers.

## 2. What the width changes

The layout, the operators and the units are BERT-Tiny's, unchanged. Four
things are new, and three of them are the same observation: **a channel is a
whole ciphertext, so anything per-channel is free here** that costs a
plaintext, a mask or a mode plan when a channel is a slot.

### (a) The feed-forward is tiled over the hidden axis

A stream is 768 ciphertexts; the 3072 intermediates the up projection makes
would be 28 GiB at the level the GELU reads them at, and their int8 split
another 22. So `up` is projected one tile of `rows_per_tile` hidden channels
at a time (ONE split of `h` serves every tile), the tile is GELU'd, and
`down` is a SEPARATE operand per tile whose products are accumulated -- the
same sum in a different order. The layer's peak is one tile.

Three smaller memory contracts, all free: `Layer` CONSUMES its input stream
rather than copying it; the attention levels `x` down to the residual's
level as soon as the projections have read it; the feed-forward does the
same to `h` once its split exists.

### (b) The GELU's interval is per CHANNEL

BERT's outliers are dimension-wise, and at this width that is not a detail:
at layer 10 one hidden channel reaches `|u| = 130` where the 99th percentile
is **5.5**. A Chebyshev interpolant's error depends on `radius / degree`
ALONE -- measured over the whole table, 0.25 gives 5e-5 and 0.125 gives
1e-12 -- so one shared interval charges all 3072 channels for the worst one:
degree 255 on +-156 is **1.0e-01 absolute**, and the host chain then returns
layer 9 at 2^-5.6, layer 10 at 2^-3.3 and layer 11 as NaN with 9555 escapes.

So sort the channels by their own `|u|` and let the feed-forward TILE -- which
already exists, for memory -- be the band. The permutation rides `wint`'s
columns and `wout`'s rows, which leaves the layer's output exactly unchanged
(a sum over the hidden axis), and each tile's `1 / a` rides the same columns
so `EvalPoly` still sees `[-1, 1]`. It is also FASTER than one interval: 11
of 12 tiles take degree 31 where they used to take 255, and only the outlier
tile pays 511.

### (c) The LayerNorm divides by H, and that is not a detail

BERT-Tiny centres for free: `centred = H x_c - sum` needs no division and
costs no level. At H = 768 it is a TRAP, and it cost this branch its first
two runs. The variance then carries `H^3 c^2 var` = 1.4e7, so the constant
that scales it back for its bootstrap is `ride / (hi / kappa0)` = 1.7e-8 --
and `Encoder::EncodeConstant` stores `BigInt(number * scale)`, so at scale
2^35 that constant IS THE INTEGER 574. Nine bits. Its relative rounding
error is 8.7e-4 = 2^-10.2, which is the 2^-10.12 the probes measured on `r`
to the digit; the same imbalance leaves the apply's per-channel constant on
`r` at 3.5e-4, where `MultScalar`'s own rescale rounding is a fifth of the
message, and the layer came back at 2^-6.9 against a host chain of 2^-19.6.

So form the mean: one `MultScalar` by 1/H on the ONE sum ciphertext, then
`centred_i = d_i x_i - mu`. Every message is then in [0.1, 20] and every
constant above 2^18, and **layer 0 goes 2^-6.28 -> 2^-11.86**. The level it
costs is taken back by reading the variance off the sums --
`H sum(x^2) - sum(x)^2` instead of `sum(centred^2)`, one level higher and a
factor H smaller -- whose cancellation is nothing here: `mu^2 / var` over the
twelve layers is at most 0.007, or 0.01 bits.

The lesson generalises: **at this width every constant has to be checked
against the scale it is encoded at.** A magnitude that is merely inelegant at
H = 128 is a nine-bit constant at H = 768.

### (d) The streams carry a public per-channel suppression

The residual stream is dimension-wise skewed too: the LayerNorm output
reaches 59.9 at layer 0 and **102 at layer 4** where a typical channel is
~1. One ride sizes every channel by the largest, so the typical channel sits
at 1/60 of the bootstrap's input height.

Each channel gets its own public factor `chan[c]`, a POWER OF TWO in
`[1, cap]`, and a ciphertext's message is `carry * value / chan[c]`:

* a projection that READS a stream folds `chan[c]` into its weight's rows,
  one that WRITES it folds `1 / chan[o]` into its columns and its bias, so a
  residual adds two streams carrying the same factors;
* a LayerNorm has to undo it before the channel sum, and that is why the
  factors are powers of two: an integer constant is encoded at scale 1, so
  `H chan[c] x_c - sum(chan[k] x_k)` is exact and costs NO level.

`cap` is a trade, not a free lunch: EvalMod's error is cubic in the message,
so lifting every channel to the full ride lifts every channel to the full
cubic term. See section 5 for what it measured.

### (e) Twelve layers do not have to be one run

`BERT_BASE_FIRST_LAYER=L` starts the chain at layer L from the float64
`h_L{L-1}.f64`: the crypto chain within the window, the exact stream at its
start. A box that must be left in half an hour still measures every layer.

The host side is chunked for the same reason: one layer's `u` for 1000
prompts is 3.1 GB and twelve layers of records would be 150, so
`reference.py` and `sim.py` walk the population a chunk at a time. Verified
to decide IDENTICALLY -- on 64 BERT-Tiny prompts the chunked sim reproduces
every window, degree and fold estimate of `bert_tiny/sim.py` to the last
digit (only the reported `fit_err` moves, at 1e-9 relative).

## 3. What is built

* `bert_base/export.py` -- checkpoint -> f32 blobs (BERT-Tiny's file with
  the default checkpoint changed; the shape comes from `config.json`).
* `bert_base/model.py` -- the float64 forward with hooks (BERT-Tiny's).
* `bert_base/reference.py` -- `h_L{k}.f64` + `stats.json`, chunked.
* `bert_base/sim.py` -- the chain on the host with the approximations in;
  picks every degree and window, and writes `calib.json`. **This is where
  every knob is DECIDED; the C++ only reads it.**
* `bert_base/debug.py` -- the tapped intermediates against the model.
* `extension/CiBertBase.*` -- `CiBertBaseLayer`; `unittest/CiBertBaseTest.cpp`
  -- the chain against `h_L{k}.f64`.
* `bert_base/vessl_{setup,host,run}.sh` -- bring-up, the host half, one run.

## 4. The knob ledger

Everything that is not a weight or an algorithm. "Who decides" says whether
the number is a THEOREM (fixed by the algebra), a CALIBRATION (from data), a
BUDGET (a level or ride bound) or a CHOICE (mine, revisable).

| knob | value (T=128, the recorded prompt) | who decides | where |
|---|---|---|---|
| ring | `ci16_35_k16_w58` (landing 16, K 16) | choice | `BERT_BASE_PARAM` |
| ride | 0.35 | budget (EvalMod cubic) | `Config::ride` |
| boot group | 8 | choice (BootBatch exact) | `Config::boot_group` |
| BSGS split | 16 x 8 (T=128) | choice | `Config::baby_steps` |
| **feed-forward tile** | 256 (12 tiles) | choice: the layer's memory peak AND the GELU's band | sim `--ffn-tile`, calib `ffn_tile` |
| **GELU per tile** | deg 511/31/.../31 at L10, intervals +-156 .. +-3.2 | calibration (per-channel `\|u\|` x 1.2) + tol | calib `gelu.tiles`, `gelu.perm` |
| **per-channel suppression** | powers of two, cap 16 | choice x budget: free everywhere but EvalMod | sim `--chan-cap`, calib `*_chan` |
| Cho passes k | 1 (oracle) | calibration: first window under `--sq-ratio` 300 | sim `--k` |
| row shift | per (head, query t) max over prompts | calibration | calib `softmax.shift` |
| exp domain / degree | L0 [-10.2, 0.25] deg 15 | calibration + margin / tol | sim `--exp-margin`, `--exp-tol` |
| fold estimate | per (pass, head, token) geometric mean | calibration (the RATIO window) | calib `softmax.est` |
| 1/sqrt windows | pass 0 [0.77, 1.3] (oracle: the fold makes it exact) | calibration + margin | sim `--margin` |
| **LN variance window / degree** | L9/L10 [0.57, 2334] deg **511** | calibration; the NARROW path is booted, so a degree is free of the wide plan | sim `--ln-max-degree` |
| tanh (head) | certified [-54.2, 54.7] deg 511 | THEOREM (the last LN's sphere) | calib `head.tanh` |
| stream carries | ride / the SUPPRESSED absmax | budget | `Prepare` |
| level plan | S at exp + 6 (k=1) or exp + 2; P >= 4 | budget (asserted) | `Prepare` |

## 5. Measured (A100, `ci16_35_k16_w58`, T = 128 x B = 512)

**The host chain (approximations only, no crypto), the recorded prompt:**

| GELU | L0 | L8 | L9 | L10 | L11 | head |
|---|---|---|---|---|---|---|
| one interval per layer | 2^-19.6 | 2^-11.4 | **2^-5.6** | **2^-3.3** | **NaN** | -- |
| per tile (12 x 256) | 2^-19.6 | 2^-18.8 | 2^-18.8 | 2^-11.7 | 2^-11.8 | 2^-14.1 |

The remaining step at layer 10 is that layer's outlier tile: 256 channels at
+-156, where degree 511 is 8.6e-04.

**On the card, layer 0** (768 wide boots + 14 narrow, 36480 rotations,
12302 relinearizations, 33 GiB, 391 MiB of GEMM operands):

| | rms vs float64 | wall | of which |
|---|---|---|---|
| no suppression | **2^-6.92** | 72.9 s | boot 32.4, GELU 12.3, scores 8.0, values 5.9, softmax 5.4, ffn 4.1, qkv 3.2, ln 2.0 |
| suppression cap 16 | 2^-6.28 | 72.9 s | (the same) |

(in progress -- section 6.)

## 6. Plan

1. B = 1, T = 128: build, layer 0. **DONE** (2^-6.92).
2. The gap between the host chain (2^-19.6) and the card (2^-6.9) is noise
   budget, not approximation, and the suppression made it WORSE, so the
   probes are running to name the stage. IN PROGRESS.
3. The 12-layer chain, in windows.
4. Population calibration on 1000 prompts, 512 held-out on the crypto.
5. The other two shapes, (256, 256) and (512, 128).
6. The head, the padding mask and serve mode.
