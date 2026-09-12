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

### (e) A wide variance window is a RIDE problem, not a degree problem

Layers 9 and 10 are the only norms in BERT-Base whose variance window is
wide: `ln2` spans **7300x and 5800x** where every other norm in the model is
under 200x. `NarrowLift` sizes the variance's message by the window's TOP, so
the smallest variance rides at `ride / ratio` and the narrow bootstrap's
ABSOLUTE error is 22 % of it. The twelve-layer chain is flat at 2^-9.6
through layer 8, **2^-6.12 at layer 9, and 2^+130 at layer 10** -- the second
being the first's consequence, since layer 10's windows were sized on an
exact layer 9 and 6 % escapes them.

A per-token public fold -- the Cho softmax's own trick -- does NOT help here:
the spread is not positional, and over 256 held-out prompts the per-position
estimate takes 4244x to 3733x. So boot the STREAM instead, for those norms
only (`Config::ln_boot_ratio`, 200). The variance is then high enough that
its inverse square root needs no bootstrap at all, so its error is the
stream's own, uniformly, whatever the window; the output lands low, which
costs nothing because both norms' outputs are bootstrapped immediately
anyway. It costs `model` wide boots on the two layers that trip it -- 3 % of
a chain. A norm that boots its input has to RIDE it, so the carry is sized on
the residual as well as the stream where the flag is set.

### (f) Twelve layers do not have to be one run

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

### The host chain (approximations only, no crypto), the recorded prompt

| GELU | L0 | L8 | L9 | L10 | L11 | head |
|---|---|---|---|---|---|---|
| one interval per layer | 2^-19.6 | 2^-11.4 | **2^-5.6** | **2^-3.3** | **NaN** | -- |
| per tile (12 x 256) | 2^-19.6 | 2^-18.8 | 2^-18.8 | 2^-11.7 | 2^-11.8 | 2^-14.1 |

The remaining step at layer 10 is that layer's outlier tile: 256 channels at
+-156, where even degree 511 is 8.6e-04.

### On the card: what each fix was worth, layer 0

| | rms vs float64 | wall |
|---|---|---|
| as first built | 2^-6.92 | 72.9 s |
| + per-channel suppression (cap 16) | 2^-6.28 | 72.9 s |
| + **the LayerNorm's mean** | **2^-11.86** | 76.8 s |

768 wide + 14 narrow bootstraps, 36480 rotations, 12302 relinearizations,
391 MiB of GEMM operands, **33 GiB** of device memory; setup 7.0 s. Of the
76.8 s: boot 32.4, GELU 12.3, scores 8.0, values 5.9, softmax 5.4, ffn 4.1,
qkv 3.2, ln 2.0.

### The twelve-layer chain, the recorded prompt in every instance

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -11.86 | -11.34 | -10.38 | -9.90 | -9.65 | -9.58 | -9.59 | -9.66 | -9.62 | -7.12 | -6.94 | -6.98 |

(rms as `2^x`; layers 9-11 measured from an exact layer 8, the others
chained.) **It is FLAT from layer 4 to layer 8** -- the noise reaches a
steady state rather than accumulating -- and then the last three layers sit
2.5 bits lower. 104 s a layer (1536 wide + 14 narrow boots), 137 s where a
norm boots its own stream.

### Where the last three layers go, probe by probe (layer 9)

`BERT_BASE_DUMP` + `bert_base/debug.py`, every intermediate against the
float64 model:

| q,k,v | s | y | sq | r | P | attn | h_pre | ln1 var | ln1 r | ln1 out |
|---|---|---|---|---|---|---|---|---|---|---|
| -13.5..-15.4 | -12.8 | -12.2 | -12.2 | -11.7 | -11.0 | -10.8 | -12.5 | -12.8 | -12.4 | **-12.0** |

so the attention and the first norm are healthy. The tail is not:

| t_gelu | **g** | y_ffn | z_pre | ln2 cen | ln2 var | **ln2 r** | **ln2 out** |
|---|---|---|---|---|---|---|---|
| -12.6 | **-8.5** | -10.6 | -10.7 | -9.6 | -10.0 | **-8.7** | **-7.1** |

Two mechanisms, both named and both with a known fix:

1. **The GELU's band is still too wide.** Tile 0 at layer 9 holds the
   channels from `|u| = 109` down to `|u| = 5.4`, and they share one `a`, so
   the ciphertext's own error `2^-12.6` in `t` becomes an ABSOLUTE
   `a * 2^-12.6 = 0.018` in `g` -- fine against the outlier's 91, four bits
   against the other 255 channels' ~1. The fix is to make the GELU's band
   independent of the memory tile and cut it GEOMETRICALLY (a band per
   octave of `|u|`, ~6 bands), so a channel never rides more than 2x its own
   scale.
2. **A 4000x variance window compresses its own polynomial's input.** Even
   with the stream booted, `t = (V kappa0 - b) / a` maps `var` onto
   `[-1, 1]` with `a = 1167`, so a token at `var = 0.57` sits at `-1 + 5e-4`
   and the ciphertext's absolute error in `t` is multiplied by `a / var =
   2047` on the way back out. The fix is the TWO-STAGE inverse square root:
   a crude `r0`, then `y0 = centred * r0` whose mean square is `(1 + e)^2`
   BY CONSTRUCTION, so the second window is a theorem near 1 and compresses
   nothing. It costs one more wide level, which the plan has at k = 1.

### The three service shapes, layers 0 and 1, each on its own oracle

| (T, B) | layer 0 | layer 1 | wall | boots a layer |
|---|---|---|---|---|
| (128, 512) | 2^-11.86 | 2^-11.34 | 76.8 / 104.5 s | 768 / 1536 |
| (256, 256) | 2^-11.91 | 2^-11.33 | 87.3 / 118.8 s | 768 / 1536 |
| (512, 128) | 2^-11.71 | 2^-11.29 | 114.2 / 151.2 s | 768 / 1536 |

The accuracy is FLAT in the shape, as the layout intends: only the diagonal
products' rotation stride and the number of score ciphertexts change, and
the extra cost is the softmax's (T ciphertexts a head).

### The population calibration, and the seven slots that ruin a batch

512 held-out prompts (a different book, real lengths 3..128) on the
1000-prompt population calibration came back at 2^-8.75 with a worst
instance of 2^-4.42, and the next layer at 2^+201. The cause is NOT the
crypto: the same chain with only the APPROXIMATIONS in, on the host,
already gives layer 0 seven escapes and a worst prompt of 2^-4.46. Naming
them exactly (`bb_esc.py`, the exact stream against its own windows):

| layer | what escapes | how far | of how many |
|---|---|---|---|
| 0 | the GELU's band | 1.170x its `hi` | 7 of 201,326,592 |
| 1 | the GELU's band | 1.001x | 1 of 201,326,592 |
| 1 | the exp's domain | | 11 of 786,432 |
| 1 | the first Cho window | 1.30x its `hi` | 3 of 786,432 |

**Seven slots in two hundred million.** That is the batched layout's
standing rule from BERT-Tiny, in its sharpest form yet: CKKS noise lives in
the coefficient domain, so one escaped slot is a batch-wide event -- and a
degree-511 polynomial outside its interval is astronomical. The margins
(`--margin` 1.3, `--gelu-margin` 1.2, `--exp-margin` 1.0) are BERT-Tiny's,
and at twelve layers of 768 channels they are simply too tight for a
held-out book.

Widened to `--margin 2.0 --gelu-margin 1.5 --exp-margin 2.0` layers 0 and 1
are clean and the card follows: **layer 0 goes 2^-8.75 (worst instance
2^-4.42) to 2^-11.34 (worst 2^-10.52)**, within half a bit of the oracle's
own. But layer 2 then explodes, and the scan over all twelve layers says
where:

| what | layers | how far over |
|---|---|---|
| the exp's domain, its TOP | 3, 4, 5, 7 | `u` reaches 0.93-1.10 where the domain tops at 0.60-0.81 |
| the first Cho window's top | 2, 6 | 25.6 vs 22.4, and **69.4 vs 29.6** |
| the GELU's band | none | worst 0.936 of its `hi` |

The exp's TOP is structural and now has its own knob (`--exp-margin-hi`):
`shift` is the population's row maximum, so a held-out row that beats it
lands above the domain by the excess over `2^k`. The first Cho window at
layer 6 is the fold's estimate not transferring -- `est` is a per-(head,
token) geometric mean of the CALIBRATION book and the held-out book's rows
sit 4.7x higher.

**This is the real lesson of the width**, and it is a calibration lesson,
not a crypto one: a served batch is `512 x 128 x 768` slots a layer and
twelve layers, so 2e8 draws -- a tail that BERT-Tiny's two layers and margin
1.3 never saw is CERTAIN here. The answer that scales is the one both the
Llama line and BERT-Tiny reached: windows that are THEOREMS (the softmax's
certified upper end, the GELU's sphere bound) rather than statistics. Until
then the margins are the knob, and they are in the ledger.

## 6. Plan

1. B = 1, T = 128: build, layer 0. **DONE** (2^-6.92).
2. The gap between the host chain (2^-19.6) and the card (2^-6.9) is noise
   budget, not approximation, and the suppression made it WORSE, so the
   probes are running to name the stage. IN PROGRESS.
3. The 12-layer chain, in windows.
4. Population calibration on 1000 prompts, 512 held-out on the crypto.
5. The other two shapes, (256, 256) and (512, 128).
6. The head, the padding mask and serve mode.
