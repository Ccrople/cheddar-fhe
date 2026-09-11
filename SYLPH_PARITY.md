# Sylph parity — what the paper specifies, and where this branch implements it

Branch `llama3_CI_Sylph`, cut from `llama3_CI` at `cc2b6e5` (the T = 4096 line).
Paper: `Sylph.pdf`, arXiv 2601.18511v2, 17 pp.

This file exists so that "we follow Sylph" is checkable rather than asserted.
Every numbered section and every displayed equation of the paper gets a row.
**A deliberate difference is a row too** — the point is not to claim parity
where there is none, and three of the rows below are places where this tree
does something else on purpose and says why.

Legend: **YES** implemented and checked · **HOST** the mathematics is
implemented and verified on the host, the encrypted half awaits a GPU ·
**DIFF** deliberately different · **NO** not implemented.

---

## 1. Section 2 — preliminaries

| § | What the paper specifies | Status | Where |
|---|---|---|---|
| 2.1 | CKKS, ring degree `N = 2^16` | YES | `parameters/ci16_35.json` |
| 2.1 | **Conjugate-invariant CKKS** [32], `N` real slots | YES | the whole CI track; `ci_ring_test`, `ci_subring_is_ci.py` |
| 2.1.1 | slot / coefficient / **SinC** encodings, eq. (1) | YES | `CiSinCBasis`, `Encoder::EncodeSinC`, `sinc_encode_test` |
| 2.1.1 | sparsely-packed ("slim") plaintexts are periodic | YES | consumed by `SlimPolyHandler` |
| 2.1.2 | bootstrapping, conversions fused into it | YES | `BootContext::HalfBootTower`, `CHEDDAR_CI_FUSED=1` (Doing 3.16) |
| 2.2 | PCMM from BLAS [27] | YES | `CoeffLinearLeg`, `PcmmBlasHandler` (int8 IMMA, Doing 3.19) |
| 2.2 | CCMM from BLAS [31], [28] batched over SinC | YES | `CiSwitchedCcmm`, `CmtTest` word-for-word |
| 2.2 | PCMv [29] / CCMv [30] for decode | YES | `CiDecodeLayer` (Doing 7.40–7.48) |
| 2.3 | SoftMax of [25] (Cho): translate, `exp`, `k` normalise-and-square passes | YES | `SoftMaxHandler`, `CiSinCAttention::SoftMax` |
| 2.3 | **the norm is EUCLIDEAN**, so no final normalisation | YES | `SoftMaxTest`'s header states it; reading it as a sum costs 19–26 levels |
| 2.3 | only the LAST inverse square root needs accuracy | YES | `iter_inv_degree` / `last_inv_degree` split |
| 2.3 | the auxiliary track runs on **sparsely-packed** ciphertexts | HOST | `SlimPolyHandler`; today the reduction broadcasts to full width |
| 2.4 | Llama-3 architecture, prefill and decode | YES | `CiLlamaLayer`, `CiDecodeLayer` |
| 2.5 | outlier phenomenon: attention sink + dimension-wise | YES | see 3.1.1 |

## 2. Section 3 — the fully-encrypted 128-token path

| § | What the paper specifies | Status | Where |
|---|---|---|---|
| 3.1.1 | sink-inducing prefix, precomputed KV, shared across queries | **DIFF** | `REF_SINKS=2` plus a public rescaled copy at every norm. **Not the paper's mechanism**: our sink rows go THROUGH the non-linearities and the paper's never do. Measured worth of the difference in "3.1.1's prefix, done properly" below |
| 3.1.1 | **orthogonal rotations fused into down-proj, o-proj, v-proj** | **MEASURED** | `quarot_export.py` (+ `QUAROT_VO=1` for the head-space `v -> o` pair, new). Tables 2 and 3 reproduced on our model by `sylph_tables.py` -- see "What the rotations actually buy" |
| 3.1.2 | precision requirement estimated by injecting modelled CKKS noise and reading perplexity | YES | `reference/scripts/sylph_precision/` — reproduced, and see the caveat below |
| 3.1.3 | non-linear degree from the calibrated range (SiLU 83 -> 31) | DIFF | `CiLlamaLayer::SiLuDegree` derives the degree from the range by a Bernstein-ellipse rule and caps at 63. See "Known gaps" |
| 3.1.3 | **bootstrap input scaling by `1/B`**, `p - log2 B` effective bits | DIFF | this tree rides at `ride / abs_max` per crossing and derives the constant (`GetMessageRatio`), which is the same idea with a per-crossing `B` instead of a global 128 |
| 3.2 | four prefill operations, formats per table 4 | YES | `CiLlamaLayer` + `CiSinCAttention` + `CiLlamaSeam` |
| 3.2 | packing layout, token index fastest, head next | YES | `AttentionPacking`, `CiSwitchedCcmmLayout` |
| 3.3 | **ring switching in SinC encoding** | YES | `CiSinCBasis`, `ci_sinc_basis_test` |
| **3.4** | **slim polynomial evaluation — lemma 1, eq. (2), eq. (3), Algorithm 1, theorem 1, appendix D** | **YES (A100)** | **`SlimPolyMath.h` + `SlimPoly.h`, new on this branch — see below.** On a card 2026-09-11: `SlimAlgorithmOne` degree 16 at `j = 4` agrees with its own plan to 2.0e-04, `SlimAppendixDFold` folds 5 levels to 4 at 1.5e-04 |
| 3.5 | end-to-end latency | n/a | a measurement, not an algorithm |

## 3. Section 4 — long heterogeneous prompts

| § | What the paper specifies | Status | Where |
|---|---|---|---|
| 4.1 | public prefill in the clear, private prefill on the encrypted tail | YES | `PublicPrefill`, `CiPcAttention` |
| 4.1 | PC-attention and CC-attention split | YES | `CiPcAttention` / `CiSinCAttention` |
| **4.2** | **eq. (5): depth-one PCMM, `tau^(l+1)(B) -> tau^l(C)`, BSGS, `O(sqrt d)` rotations** | **YES (A100)** | **`SylphPcmmMath.h` + `SylphPcmm.h`, new on this branch.** On a card: 3.5e-06 = 17.97 bits at `d = 32`, 10 rotations, ONE level -- after the first run's 0.999 relative error found the layout contract: the matrix must be TILED with period `d^2` when `d^2` is less than the slot count (`SylphPcmm::Apply`) |
| 4.2 | `pt_{A,i,j,l}` rebuilt at runtime from one stored `tau^l sigma(A)` | HOST | `sylph_pcmm::PlaintextFor`, the `cache_plaintexts` argument |
| 4.2 | `tau^2` applied once, right after RoPE | HOST | `sylph_pcmm::TauPermutation` + `SlotPermute`; one level, and **64 diagonals at `d = 128`, measured** -- an even power halves the orbit of `n j mod d`, so the one the paper applies is the cheapest |
| **4.3** | **SoftMax on `tau(M)` is `tau(SoftMax(M))`** | **HOST** | verified as an identity in `SylphPcmmTest.SoftMaxSeesColumnsOfTau` |
| 4.3 | two Cho iterations, 8 levels in the main track | YES | `SoftMaxCalibration::niter`; at our `m_eff` **k = 1 is better** and the header says why |
| 4.3 | sharp range estimates from distributional calibration | DIFF | **this is the one we changed on purpose**; see below |
| appendix E | lemma 2 | HOST | `SylphPcmmTest.LemmaTwo` |

## 4. Section 5 — the multi-GPU runtime

Not applicable: this tree targets one GPU per layer and its own scheduling
(`SylphSchedule`, `CiLayerPrefetch`). Section 5's data partitioning,
inter-GPU communication and model conversion are a system, not an algorithm.

---

## The new code, and what was verified without a GPU

Both of the headline additions are split into a pure-host half and an
encrypted half **on purpose**: the mathematics is where these algorithms can be
wrong, and it can be checked before a card is asked for.

    slim_poly_test  --gtest_filter='SlimMath.*'        no GPU
    sylph_pcmm_test --gtest_filter='SylphPcmmMath.*'   no GPU

### Section 3.4 — slim polynomial evaluation

`include/extension/SlimPolyMath.h`, `src/extension/SlimPolyMath.cpp` (the
decomposition), `include/extension/SlimPoly.h`, `src/extension/SlimPoly.cpp`
(Algorithm 1), `unittest/SlimPolyTest.cpp`.

Verified on the host: lemma 1 on polynomials whose answer is known by hand
(`m = 2` exactly, residual 8.9e-16); the eq. (3) tree against the polynomial it
claims to evaluate at degrees 16 / 32 / 64 and every depth `j` (residual
3.2e-14 / 3.9e-13 / 1.3e-8 against fit errors 2^-1.2 / 2^-5.7 / 2^-14.3);
theorem 1's level count; the negated case.

**Appendix D is the load-bearing part**, and it is the measurement worth
carrying forward. It asks for two things — `U` and `V` small on the interval,
and children whose own `m` is small. Implementing only the first, which is the
half that is easy to read, puts `m` at **8.8e5** on the softmax's own inverse
square root window at degree 64: 16.2 bits of dynamic range spent on the
cancellation `U^2 + V^2 - m` that a direct evaluation never pays. With the
lookahead, `m` stays at **1.14** and the cost is 0.1 bits, at every depth. So
`DecomposeOptions::lookahead` defaults on and `SlimPlan::range_bits` is
reported beside the residual.

**What it buys, stated once** (`slim::Budget`, `SlimBudget`): at `L` levels
Paterson-Stockmeyer buys degree `2^L - 1` and slim buys `2^(L-1)`, so slim is a
SPEED technique and never an accuracy one — *unless* appendix D's
leading-coefficient fold is used, and then it buys `2^L` at the same levels and
dominates on both axes. Measured on `[1/128, 1]`, at 6 levels: PS degree 63 in
16 multiplications against slim+fold degree 64 in **7 multiplications and 3
rotations**.

Deliberately not done: the leaf is the whole Chebyshev basis,
`O(2^(k-j))` multiplications rather than Paterson-Stockmeyer's
`O(2^((k-j)/2))`. It costs nothing where appendix D's search makes slim usable
— `j = k` and `j = k-1` leave a leaf of degree 1 or 2 — and the refinement is
`EvalPoly`'s tree with `PAccum` in place of `CAccum`, which is the same change
the BERT branch's mode plan wants.

### Section 4.2 — the depth-one PCMM

`include/extension/SylphPcmmMath.h`, `src/extension/SylphPcmmMath.cpp`,
`include/extension/SylphPcmm.h`, `src/extension/SylphPcmm.cpp`,
`unittest/SylphPcmmTest.cpp`.

Verified on the host: lemma 2's two identities at `d = 4, 8, 16` and every `k`;
JKLS eq. (4) reproducing `A B`; **eq. (5) reproducing `tau^l(A B)` at every
`l` in 0..3, every legal baby-step split, and `d` up to 32**; the cost claim
(`d = 128` gives `b = 8, g = 16`, 22 rotations against a naive 127, and one
level); and section 4.3's SoftMax identity.

The index that has to be right, and the reason a host test exists at all, is
`pt_{A,i,j,l} = rot_R^(-l(i + j b) - j b) . rot_C^(i + j b) (tau^l sigma A)`:
one shift comes from lemma 2 each time `tau` passes a column rotation and the
other from hoisting the giant step out of the inner sum. Getting either wrong
gives a permutation of the right answer, not an obviously broken one.

---

## Known gaps and deliberate differences

**1. The SiLU's degree and range (3.1.3).** The paper drops SiLU to degree 31
on a calibrated range of 10.82. `SiLuDegree` computes what its own rule asks
and caps at 63; at layer 31's measured range of 41 the rule asks for 141, so
the cap is 2^-5.2 and that is the pipeline's current limiter (Doing 3.27).
The range is also a corpus statistic (`1.2x` an observed maximum) that 7 of 600
held-out prompts left.

**And it is worse than that, which the measurement settled.** The certified
sphere bound `|g_j| <= sqrt(H) ||gain . W[:,j]||` is exactly invariant under
QuaRot, because the rotated model has `W' = Q (gain . W)`, `gain' = 1`, and
`||Q v|| = ||v||`. But the same algebra kills the *observed* range too:
`g' = (u Q) . (Q w) = u . w = g`, so the SiLU's input is **the same number**,
not merely the same bound. Measured over 8 held-out Wikipedia prompts and all
32 layers, the SiLU range is identical to three digits at every single layer —
layer 31 is 29.51 with the rotation and 29.51 without. See "What the rotations
actually buy" above.

**2. `first_hi` is a theorem here, not an estimate (4.3).** Section 4 says
"Rather than relying on worst-case bounds as [25], we use the distributional
data computed during calibration to obtain sharp estimates on the range of the
inverse square root computations." This tree did that too, and 600 held-out
articles broke it: 12/600 escaped the first window and 10 prompts returned
non-finite values. The fix (Doing 3.27) is that the first window's upper end is
**derivable**: `beta = 4 hb / span_raw = 2^(1-k)/sqrt(D)` — the span cancels —
so with a row-shift margin making `rs >= the row's own max`,
`sq_0 <= live_i exp(beta (esc - a span)) / est_i` with `live_i = i + 1` public.
Measured 2^-4.042 -> 2^-5.519, escapes 12/600 and 9/600 -> 0/600 and 0/600.
**This is a deliberate divergence from the paper's stated method**, and it costs
0.37 bits at degree 15 and nothing at 31/63.

**3. Bootstrap scaling (3.1.3).** The paper scales a ciphertext by `1/B` with a
single global `B = 128` and accepts `p - log2 B` bits. This tree rides each
crossing at `ride / abs_max` with the constant derived rather than fitted
(`GetMessageRatio`, `CHEDDAR_CI_RIDE`), which is the same trade with a
per-crossing `B`. Not a gap, but not the same number either.

**4. Section 3.1.2's protocol does not test what breaks.** The paper's accuracy
evidence is table 7, produced by injecting modelled CKKS noise into
*exactly evaluated* non-linearities. That protocol cannot see a polynomial
evaluated outside its interval, which is the failure mode of items 1 and 2
above and the one that produced 10 non-finite outputs in 600 prompts here. The
paper reports no accuracy number measured from the encrypted execution.

---

## What the rotations actually buy, measured

`reference/scripts/sylph_tables.py`, 8 held-out Wikipedia prompts of 128
tokens, all 32 layers, variants applied IN MEMORY so nothing depends on a
second export agreeing with the first. `sink` is what this tree ships today;
`vo` adds the residual rotation and the head-space `v -> o` pair.

| quantity | `sink` | `+ rotation` | factor | [SYLPH] |
|---|---|---|---|---|
| SoftMax input | 76.88 | **76.88** | **1.00** | 39.24 -> 32.78 |
| SiLU input | 29.51 | **29.51** | **1.00** | 23.00 -> 10.82 |
| RMSNorm input | 29.31 | 3.67 | 8.0 | 2243.97 -> 7.65 |
| down-proj output | 295.00 | 30.39 | 9.7 | 310.56 -> 1.92 |
| o-proj output | 8.32 | 1.17 | 7.1 | 10.12 -> 1.08 |
| v-proj output | 9.01 | 5.63 | 1.6 | 5.89 -> 4.74 |

Three things follow, and the first is the one that matters.

**1. The rotation cannot touch the SiLU or the SoftMax, and that is algebra,
not a sampling artefact.** Their inputs are projections of a NORMED vector, and
a residual rotation cancels inside the projection: `g' = (u Q)(Q w) = u w = g`.
The per-layer table confirms it to three digits at all 32 layers (layer 31:
29.51 against 29.51; only the `none` column moves, and only in the third digit,
which is the sink rescaling). **The SiLU is this pipeline's current limiter, so
3.1.1 buys our limiter nothing.** The 0.01-bit verdict that turned QuaRot off
at 1.5dv was right, and now the reason is known — it was never about layer 2
being unrepresentative.

**2. What it does buy is real, and it is everything that LIVES in the residual
stream**: RMSNorm's input 8x, down-proj's output 9.7x, o-proj's output 7.1x.
Those are the operators the rotation reaches, because their input or output IS
the rotated space rather than a projection out of it.

**3. The head-space `v -> o` rotation is needed for table 3's third row and
nothing else does it.** `resid` alone leaves v-proj at 9.01; `vo` brings it to
5.63. That is why §3.1.1 names three layers, and it is the piece
`quarot_export.py` did not have.

**Caveats, because these are maxima.** Eight prompts is a weak estimate of a
maximum and more prompts can only raise it. And our `none` column is NOT
[SYLPH]'s "Baseline": it already carries the BOS x 2 prefix in the token ids
and differs from `sink` only by the public rescaling. That matters for one
comparison in particular — the paper attributes SiLU 23.00 -> 10.82 to
*prefixing*, and our prefixing does nothing for the SiLU (29.44 -> 29.51).
[SYLPH]'s prefix is a chosen-token KV cache identified offline; ours rescales
the magnitude of two BOS tokens. **Those are different mechanisms, and theirs
is the one reported to move the SiLU.** That, not the rotation, is the lead
worth following for the remaining limiter.

---


## 3.1.1's prefix, done properly, and what is actually left

`sylph_tables.py` closed the rotation half of 3.1.1 (it cannot move the SiLU or
the SoftMax at all). `sylph_prefix.py` does the prefix half, and the difference
between the two prefixes is a mechanism, not a parameter:

    ours     BOS x 2 inside the prompt plus a per-token rescaling at every
             norm. The sink rows GO THROUGH every non-linearity.
    [SYLPH]  the sink tokens' KV precomputed offline and injected as a STATIC
             KV cache. Those rows are public and never reach the encrypted
             non-linearities at all.

The sink rows are prompt independent by causal masking -- row `t` sees keys
`0..t` -- which is what makes the paper's design legal and what this tree
already relies on for `attn_sink` / `ffn_sink` / `up_sink`. So the measurement
is: the same forward, maxima taken over the USER rows only.

**SiLU input range, all rows -> user rows** (4 held-out prompts, 32 layers):

| L | all | user | |
|---|---|---|---|
| 0 | 4.11 | **2.60** | |
| 1 | 15.25 | **4.05** | 3.8x; reproduces `CiLlamaLayer.h`'s independent "15.25 at the two sink rows against 3.71 at the 126 user rows" |
| 2-28 | 3.5-10.1 | unchanged | already comfortable |
| 29 | 14.53 | **9.51** | |
| 30 | 18.23 | **10.56** | |
| **31** | **29.51** | **29.51** | **does not move** |

`rmsnorm_in` tells the same story from the other side: 227.42 over all rows
without the rescaling and **27.88 over user rows either way**. The norm
rescaling exists purely to bring the SINK rows down; take them out of the
ciphertext and it has nothing left to do.

`sink_mass` -- the mean SoftMax mass landing on the prefix keys -- is **0.93**
for BOS x 2 already, so prefix SEARCH is not where the remaining range is.

**By `SiLu.h`'s own measured table (degree 63 reaches 14.6 bits at +-24), every
layer but 31 is comfortable at the shipped ladder and layer 1 would drop from
degree 63 to 19, which is a LEVEL.**

### Layer 31 is the whole remaining limiter, and it is a weight property

The SoftMax span (which IS `m_eff`) is 16-26 at most layers, 44.94 at layer 0
and **78.23 at layer 31**. That half is already solved, and not by prefixing:
`gen_b1_pop.py` raises the Cho count PER LAYER until the first window fits,
and its own comment names layer 31 as the reason ("m_eff 98.6, hb 12.3 at
k = 2 ... one more Cho pass halves `hb` and the window collapses").

The SiLU half is not, and cannot be, because it is in the WEIGHTS. The
certified sphere bound `sqrt(H) ||gain . W[:,j]||` per channel:

| L | max | p99 | p90 | p50 | max/p50 |
|---|---|---|---|---|---|
| 30 | 81.9 | 67.0 | 49.7 | 31.1 | 2.63 |
| **31** | **168.4** | 61.4 | 47.5 | **29.2** | **5.77** |

Every other layer is 2.4-4.2. Layer 31's MEDIAN channel is layer 30's; its tail
is not. A handful of outlier channels set the range, and **no prefix moves a
weight**.

The fix is a CERTIFIED per-channel plan -- the BERT branch's banded / mode GELU
plan (`GeLuHandler::ApplyModes`) on Llama's SiLU. Priced at layer 31, RMS over
channels against |SiLU| ~ 30:

| plan | RMS | |
|---|---|---|
| shipped, statistical `1.2 x 29.5 = 35.4`, d63 | 2^-11.1 | 7/600 prompts escape |
| the same at the 600-prompt max `1.2 x 41 = 49` | 2^-8.8 | 7/600 escape |
| certified per-channel, d63 | 2^-9.5 | **escape-proof** |
| certified per-channel, d127 | **2^-12.3** | **escape-proof**, one more level |

**That table is layer 31 only, and pricing all 32 corrects it**
(`silu_plan.py`). The certified bound is 3 to 7x the OBSERVED range at most
layers -- layer 2 observes 3.53 against a bound of 31.3 -- and a Chebyshev
error is uniform in absolute terms over its interval, so a certified interval
throws away exactly that ratio. Worst layer over the whole chain, which is what
a chain reports:

| plan | worst layer | |
|---|---|---|
| shipped, statistical, d63 | 2^-11.0 | + escapes |
| prefixed rows, statistical, d63 | 2^-11.0 | + escapes |
| 8 certified bands, d63 | **2^-7.7** | a 3.3-bit REGRESSION |
| 8 certified bands, d127 | **2^-12.0** | escape-proof |

Layer 31 looked like a wash at d63 only because its observed range (29.51)
happens to sit at its bound's p50 (29.2); nowhere else does. **So certification
costs one level** -- and that is the looseness of the bound, not the band
count: more bands do not help, because the widest band still spans the bound's
maximum.

Which plan wins also depends on how many prompts the statistic saw, because the
bound does not move and the statistic does:

| layer 31 | 4 prompts | 600 prompts (measured) |
|---|---|---|
| shipped, d63 | 2^-11.0 | 2^-8.8, 7/600 escape |
| certified 8 bands, d127 | 2^-12.0 | 2^-12.0 |

The remaining prize is the last column either way: a BOUND CANNOT BE ESCAPED,
and an escape is not a small error -- outside its interval a degree-63
Chebyshev is `cosh(63 arccosh v)`.

The band curve flattens at 8 (RMS at layer 31, degree 127: one band 0.368, four
0.00986, **eight 0.00743**, thirty-two 0.00626, per-channel 0.00587) -- so eight
bands take 5.63 of the 5.97 bits a per-channel plan would give, and a band's
mask is a per-CHANNEL plaintext, the shape RMSNorm's weights already are.

**Two independent routes agree on 2^-12, and only one arrives clean.** The
600-article audit's `fix_silu` variant -- the SiLU ladder extended, range still
statistical -- measured the whole chain at 2^-12.043 with 3 non-finite outputs
left. The certified 8-band plan at degree 127 prices at 2^-12.0 and cannot
produce those three, because they are escapes.

### The prefix axis, closed

Seven prefixes, 4 held-out prompts, ranges over user rows:

| prefix | `sink_mass` | L31 SiLU | L31 span |
|---|---|---|---|
| `[BOS, .]` | 0.9126 | **27.47** | 80.92 |
| `[BOS]` | 0.9126 | 27.59 | 80.91 |
| `[BOS, \n]` | 0.9136 | 27.79 | 80.88 |
| `[BOS x 2]` (shipped) | 0.9326 | 29.44 | 78.29 |
| `[BOS x 2]` + rescale | 0.9342 | 29.51 | 78.23 |
| `[BOS x 4]` | 0.9483 | 28.95 | 75.32 |
| `[BOS x 8]` | 0.9619 | **31.91** | **72.54** |

**The SoftMax span responds monotonically to absorption** -- 0.913 -> 0.962
takes it 80.9 -> 72.5, about -10 %, which is the same size of effect [SYLPH]
table 2 attributes to prefixing (SoftMax 39.24 -> 32.78, -16 %). So the
mechanism is real and reproduced. It is also useless to us: the SoftMax is
already handled by the per-layer adaptive Cho count, and 10 % changes nothing.

**The SiLU moves the WRONG WAY** -- the most absorbing prefix (`bos8`, 0.962)
is the worst at 31.91 and the least absorbing (`bos1_dot`, 0.913) the best at
27.47 -- and the 16 % spread is the sampling noise of a maximum over 4 prompts,
which the layer 24-30 columns jitter by just as much. Changing the token (`.`,
newline) changes nothing either.

So: length, token and absorption all fail to move layer 31, on a quantity that
needs a factor of three. The prefix axis is closed.

### The prefix search on a GPU: sixty prefixes, 528 prompts, a held-out split

2026-09-11, on an A100. `sylph_prefix_gpu.py` is the port: the model resident
on the card, per-PROMPT maxima recorded, and it agrees with `sylph_prefix.py`
to **3.3e-06 on every quantity at every layer** (fp32 accumulation order).
1.2 s a variant at 4 prompts where numpy took ~9 min; 27 s at 528 prompts in
TF32, which moves no recorded quantity by more than 8.3e-04.
`prefix_heldout.py` sets every interval on 11 of 22 Gutenberg books (264
prompts, `gutenberg_ids.py`: evenly spread windows, rows interleaved by book,
no BOS in the ids) and serves the OTHER 11 -- different documents, which is
the question a statistical calibration has to answer. Everything is in
`reference/audit/2026-09-11_prefix_gpu/`.

**The sink rows are prompt independent, bit for bit, at all 32 layers.**
`SINKCHECK=1` runs one prefix over two disjoint prompt sets and compares the
sink rows' gate: relative difference **0.000** for `[BOS]`, `[BOS x 2]`,
`[A]` and `[BOS, A]`. This is the fact the static KV cache rests on, and it
had only ever been checked on layer 0. (The host emulation's per-token
rescale moves them by 1.7e-03, because it reads its target off the batch's
user rows; the crypto's factors are calibration constants.)

**BOS at position 0 is not optional.** Every prefix that starts with BOS
leaves the user rows' attention-norm window at ~11; every prefix that does
not puts it between 1,065 and 538,947:

| prefix | L1 SiLU (user) | attn-norm window (user) | L31 SiLU (user) | L31 span |
|---|---|---|---|---|
| `[BOS x 2]` (shipped) | 5.82 | 11.3 | 40.90 | 84.55 |
| `[BOS] + "Text:"` (best BOS-first at L31) | 5.75 | 11.1 | 38.48 | 87.18 |
| `[BOS x 8]` | 4.86 | 20.8 | 42.46 | 80.60 |
| `["A"]` | 20.99 | 121,574 | 57.05 | 87.31 |
| `["\n"]` | 16.35 | 284,958 | 105.64 | 87.66 |
| `["A" x 8]` | 23.26 | 1,077 | 31.44 | 87.80 |
| none | 21.67 | 538,947 | 38.97 | 90.52 |

Without BOS the model builds its attention sink on a USER token, at a
position that depends on the prompt, so no public per-token factor can take
it out -- and the invsqrt window the ciphertext would see goes from 11 to
four to five orders of magnitude more. Every single token tried (`A`, `The`, ` the`, `.`, `,`, `\n`,
`:`, `=`, ` `, `0`, `I`, `a`, `#`, `!`, `?`, `-`, `*`, `|`, `"`, `(`, `<`,
`...`, `Hello`, and five Llama-3 special tokens) does this. `["A" x 8]`'s
31.44 at layer 31 looks like the best number in the study and is not: its
norm window is 1,077 and its layer 1 is 23.

**Among BOS-first prefixes nothing moves layer 31.** Twenty-one of the
twenty-two span 38.5 to 42.5 there (`[BOS x 4]` 47.5 the one outlier), and two disjoint
halves of ONE prefix's prompts already differ by 7 % (`[BOS x 2]`: 40.90 on
the calibration books, 38.3 on the served ones). The spread is the sampling
noise of a maximum. The SoftMax span is the quantity a prefix does move,
monotonically with absorption as before: `[BOS x 8]` 80.6 < `[BOS x 4]` 82.4
< `[BOS x 2]` 84.6 < non-BOS ~87-88 < none 90.5 -- and the longer BOS runs
pay for it in the norm window (20.8 at x 8).

**And the static-KV accounting survives the held-out test.** Over the rows a
static KV cache leaves the ciphertext, the SiLU's calibrated range at
`[BOS x 2]`, with the served books' escapes past `1.2 x` it:

| L | all rows | user rows | served escapes |
|---|---|---|---|
| 0 | 3.93 | 3.46 | 0 |
| 1 | 15.59 | **5.82** | 0 |
| 29 | 14.55 | 12.63 | 0 |
| 30 | 18.24 | 12.27 | 0 |
| 31 | 40.90 | 40.90 | 0 |

The four-prompt study's layer 1 (15.25 -> 4.05) was the right shape; on 264
calibration prompts it is 15.59 -> 5.82, and no prompt from another book leaves
the user interval there. The statistical interval does still escape
elsewhere -- 3 of 8,448 served prompt-layers at `[BOS x 2]`, one of them at
1.86x its calibration maximum (layer 10) -- which is the case for the
certified bands, not against the prefix.

### What was implemented, and the one thing that was not

`gen_b1_pop.py` gains `SILU_SINK=1`: it measures `gate_absmax_user`, derives a
public per-token factor `s` that brings the sink rows inside the user interval,
and writes the public correction `SiLU(g) - SiLU(g s)` per (sink token,
channel). The identity `y = (SiLU(g s) + restore) up` is exact at every row,
`s` rides the gate crossing's existing multiply and `restore` is a plaintext
add, so it costs no level. It also CHECKS that the sink rows' gate really is
prompt independent rather than asserting it. Default off.

**Written 2026-09-11 on the module basis, and exact on the card** -- see "On
the card" below. What follows is why it waited for one. `s` is a
per-token factor and `CrossingPlaintext` already handles exactly that shape,
but `restore` is per (token, CHANNEL), and the half-density banded convention
places a channel's duplicate one token position BACK from its live copy --
`CrossingPlaintext`'s own comment records the bug that convention produced and
its symptom ("live 2^-5.99 against duplicate 2^-1.06 and a layer at relative
265"). Writing a 2D plaintext into that convention without a card to check it
against is how that bug comes back. The module basis has no duplicate band and
is the easier half; the mapping still has to be read off a running layer first.

And the cleaner design, which the measurement argues for, is not "suppress and
restore" at all: if those rows are public, they should not be in the ciphertext
in the first place. That is [SYLPH]'s static KV cache, it is a small instance
of the section 4 machinery this branch already has (`CiPcAttention`), and it
would retire all three of `attn_sink`, `ffn_sink` and `up_sink` together with
their calibration.

### On the card (2026-09-11, A100-80GB, `ci16_35` + `land13c2e9` + `land17c3e10`)

The oracle calibration (`reference_forward.py`), the recorded module-basis
fused configuration, one layer per run, every A/B pair on ONE
`CHEDDAR_RNG_SEED` -- new in `Random.h`, and checked first: the same run twice
prints the same LAYER line to every digit (2^-6.31746 both times), so a pair's
difference is its knob and nothing else. Logs in
`reference/audit/2026-09-11_prefix_gpu/crypto/`.

**The regression holds.** Layer 0 on the defaults: **2^-6.291 / rms 2^-6.457**
(recorded 2^-6.31 / 2^-6.46). Nothing this branch added moves the default path.

**The sink plan at the SiLU is exact** (`silu_sink_inject.py` over the oracle;
`Calibration::silu_sink` + `silu_restore`, the restore a per-(token, channel)
plaintext on the module basis):

| layer | knob | range | user rows | sink rows |
|---|---|---|---|---|
| 1 | off | 18.30 | 2^-6.244 / rms 2^-6.356 | 2^-7.07 |
| 1 | **on** | **4.46** | 2^-6.260 / rms 2^-6.353 | 2^-7.13 |
| 30 | off | 21.88 | 2^-7.665 / rms 2^-7.476 | 2^-9.58 |
| 30 | **on** | **12.06** | 2^-7.648 / rms 2^-7.475 | 2^-9.59 |

The sink rows land where they should (a restore at the wrong address would
show there and nowhere else, which is why the test now prints them), and the
closing numbers do not move -- because these layers sit on the crypto floor,
not on the SiLU. What the plan buys the SiLU itself, priced in double with the
crypto's own polynomials (`sink_silu_sim.py`, the FFN's error alone):

| L | today | sink plan | FFN out |
|---|---|---|---|
| 0 | 4.92, deg 31 | 2.53, deg **15** | 2^-22.7 -> 2^-19.0, **one level freed** |
| 1 | 18.30, deg 63 | 4.46, deg **31** | 2^-10.4 -> **2^-25.6**, **one level freed** |
| 29 | 17.44, deg 63 | 11.67, deg 63 | 2^-15.9 -> 2^-23.6 |
| 30 | 21.88, deg 63 | 12.06, deg 63 | 2^-12.3 -> 2^-23.5 |
| 31 | 22.08, deg 63 | 18.21, deg 63 | 2^-11.8 -> 2^-14.9 |

So [SYLPH]'s prefix, done properly at the SiLU, is 3 to 15 bits on the SiLU and
a level at two layers, for no level and one plaintext add. It shows in a
closing number only where the SiLU is the limiter -- the held-out calibration's
statistical interval, not this oracle.

**The certified bands are CORRECT and LOSE at layer 31.** Same seed:

| layer 31 | on the card | the same layer in double (`band_l31_sim.py`) |
|---|---|---|
| one interval (22.08), deg 63 | 2^-3.342 / rms 2^-4.972 | fit alone 2^-11.82 |
| one interval, **deg 127** | 2^-3.342 / rms 2^-4.972 | fit alone 2^-25.60 |
| 8 bands, deg 63 | **2^-0.72** / rms 2^-1.89 | 2^-0.67 / rms 2^-1.88 |
| 8 bands, deg 127 | **2^-2.61** / rms 2^-3.71 | 2^-2.54 / rms 2^-3.82 |

The double reproduces the card to a tenth of a bit, so the circuit is right
(the `Rescale ... in-place` warnings the band path prints are benign) and the
PLAN is what costs the bits. One band carries all of it: the top band's five
channels, certified to 168.35 and reaching 15.17 on the served prompt -- an
11x overestimate -- with the layer's largest `|u|` (14.63). And layer 31 is the
worst place to be imprecise: its FFN output (214) cancels the residual's
massive activation (217) down to `|h| <= 24.7`, so the FFN's relative error
arrives multiplied by ~8. `silu_plan.py` priced the bands as an RMS over
channels against `|SiLU|`, which sees neither the maximum, nor `u`, nor the
cancellation; its "8 bands, deg 127, 2^-12.0 worst layer" is WRONG at layer 31
and is withdrawn. The degree-127 single interval also settles the level
question: it compiles and runs in the FFN ring's turn (`1 + 7 + 1 = 9` of 9).

---

## Per-layer degrees on a B200 (2026-09-11): what the level budget affords, operator by operator

The question asked: the non-linearities' input ranges differ by layer, so let
the DEGREE differ by layer -- layer 31's SiLU alone wants more -- and where a
degree cannot be raised, say what stops it. Measured on the runpod B200
(sm_100, CUDA 13, `build100`/`build100b` of `5c210ae` + this session's
edits), B = 1, T = 128, `[BOS x 2]`, the recorded rings (`ci16_35` +
`land13c2e9` + `land17c3e10`, module basis, fused leg), one seed for every
A/B (`CHEDDAR_RNG_SEED=11`). Logs and tables in
`reference/audit/2026-09-11_b200_degrees/`.

### The rules already pick a degree per layer; the CAPS were the literals

`SiLuDegree(range)`, `NormDegree(window)` and `ExpDegree(hb)` derive each
layer's degree from its own calibration. What was per-layer only in name was
the cap: 63 / 15 / 15 typed beside the rule. They are level budgets, so they
are computed as budgets now (`CiLlamaLayer.cu`), against the FFN ring's slot
level `op = 12` and StC level `StC = landing - slack`:

| operator | the arithmetic | rungs | budget at slack 9 (StC 4) | at slack 10 (StC 3) |
|---|---|---|---|---|
| SiLU (FFN ring) | tree `Log2Ceil(d+1) <= op - 1 - StC` (the gate multiply needs one, ToCoeff wants it at or above StC) | 15 / 31 / 63 / **127** / 255 | 7 levels -> **127** (the old cap, 63, was one level SHORT) | 8 -> 255 |
| RMSNorm invsqrt (both norms, FFN ring) | square + scale (2), tree, weight multiply with NO rescale one below, ToCoeff wants StC + 1: `Log2Ceil(d+1) <= op - 4 - StC` | 9 / 15 / **31** / 63 | 4 -> 15 (= the old cap) | 5 -> 31 |
| softmax exp, single pass (tower `top` 16, `forward_level` 3) | exp tree + invsqrt tree `<= exp_in - 3 - (forward + 2) = 7`; a 4-level exp (15) leaves the invsqrt 3 (7) | 7 / 9 / 15 / 31 | (15, 7) EXACTLY; (31, 3) also fits and LOSES (below) | no ring has top 17 |
| softmax, Cho path (`niter` k) | first invsqrt at `poly_in` 8 with floor 3 -> `<= 31`; last at `top - 2` = 14 with floor 5 -> `<= 255`; exp reads `hb = m_eff / 2^(k+1)`, so **k is the lever** | k per layer | 31 / 63 shipped | -- |
| EvalMod K (the boot's range) | the ModRaise wrap-around, i.e. the secret's l1 norm IN THE CROSSING'S BASIS -- not the scale, not q0 | K 16 / 32 / 64 = 8 / 9 / 10 EvalMod levels | native 11.6 -> K 16 (the score Boots); module 20.0 -> K 32 (FFN); tower 32.1 at h 16, 49.7 at h 32 -> K 64 (leg) | -- |

Slack 10 runs (`CHEDDAR_CI_FFN_SLACK=10`, StC 3, coefficients at level 1):
layer 0 2^-6.32 / rms 2^-6.46 (= slack 9), layer 1 **2^-6.26 / 2^-6.35** with
SiLU 127 and the attention invsqrt at **31** (window 12.7; = the recorded
2^-6.24 / 2^-6.36), layer 31 2^-3.65 / 2^-4.95. So the extra level EXISTS
and costs nothing visible; nothing on the card wanted it either.

What the rules choose on the oracle calibration (`degree_plan_oracle.txt`):
SiLU 127 at layers 1, 30, 31 (ranges 18.3, 21.9, 22.1; 63 elsewhere or
lower), the norm's 31 at layers 1 and 2 only and only at slack 10, and the
single-pass softmax pair (31, 3) at layers 0 and 31 only.

### The 2^42 family cannot serve this layer, and K is why

`ci16_42_k{16,32,64}_w60` measure p = 20.93 / 19.61 / 18.28 against the
recorded rings' ~15 (`CI_PARAM_20BIT.md`), and land anywhere from 0 to
13 / 12 / 11 at that p. Three things stop them here, in order of hardness:

1. **The keyless crossing needs the same primes.** Ciphertexts cross the
   three rings at shared levels because levels `0..L` are `ci16_35`'s primes
   (`CiModelTest.cpp:467-471` asserts it). A 2^42 pool's primes are not, so
   the FFN ring cannot move alone; all three rings move or none. (The
   `k32ffn_L0` run in `sylph_suite.sh` stopped one step earlier, on the
   documented build-tree trap -- `cannot open .../unittest/ci16_42_k32_w60.json`
   -- the JSONs are copied into both trees now; the assert was not re-run.)
2. **The leg's levels.** The tower needs landing 17 and three CtS levels
   (k64 lands 11 with two); the score Boots need 16 on the base ring (k16
   lands 13). Every missing level is 42 bits of logQ: k16 at 1715.6 + 3
   levels ~ 1842, k64 at 1728.0 + 6 levels + a CtS level ~ 2050, against
   the 1771.06-bit security anchor every ring here sits under.
3. **K.** The wrap-around is a property of the SECRET's norm in the basis the
   crossing reads (`ci_module_basis.py` check 6, `ci_nested_sinc.py`
   check 5, `wraparound_K.txt`): native-sparse in native coordinates 11.6
   (K 16 holds), module-sparse in module coordinates **20.0** (K 16 does not,
   32 does with 1.6x), tower-sparse in tower coordinates **32.1 at h = 16,
   49.7 at h = 32** (K 64), and any secret not sampled in the crossing's
   basis 270-900. So the K-16 pool -- the precise one -- can serve neither
   the FFN nor the tower at any scale, q0 does not enter (the wrap divides
   by it), and the only knob is h, which is security: the least secure h
   tried still needs K 32 / K 64. A double angle buys K x2 for one level and
   ~1.2 bits, which is exactly the two levels K 64 costs over K 16 -- that is
   the whole price and it is not negotiable. The native boot IS K-16-friendly,
   and going back to it means the converter route the module and tower bases
   were built to retire.

### On the card, oracle calibration: three chains and the single-layer A/Bs

All 32 layers, ledger on, seed 11. `base32` and `deg32` ran while the
simulator held the card too, so their times are not the layer's.

| chain | L0 | L1 | L18 | L30 | L31 | worst | median | boots |
|---|---|---|---|---|---|---|---|---|
| `base32`: SiLU <= 63, invsqrt <= 15, exp 15 / 7 (the recorded rule) | 2^-6.33 / -6.40 | -6.17 / -5.96 | -4.84 / -5.80 | -5.31 / -6.07 | **-3.78 / -5.34** | L31 2^-3.78 | 2^-5.71 | 3072 |
| `deg32`: the SiLU budget ladder (127 at L1, L30, L31) | -6.33 / -6.40 | -6.19 / -5.96 | -4.83 / -5.79 | -5.33 / -6.03 | -3.72 / -5.25 | L31 2^-3.72 | 2^-5.70 | 3072 |
| `exp32`: + the softmax (exp, invsqrt) pair solver | -6.44 / -6.46 | -6.24 / -5.97 | -4.82 / -5.77 | -5.45 / -6.02 | **-2.95 / -4.22** | L31 2^-2.95 | 2^-5.67 | 3072 |

Single layers from the clean stream, seed 11: `exp_L0` 2^-6.41 / -6.51,
`exp_L30` 2^-7.64 / -7.47, `exp_L31` **2^-2.93 / -3.88** with its seam at
**2^-1.48** (the recorded seam there is 2^-5.63).

Read together:

* **The SiLU's per-layer degree is right and free, and invisible here.** 127
  where the range asks (L1, L30, L31) changes no closing number by more than
  the seed's own spread, because those layers sit on the crypto floor -- the
  same finding as the sink plan's. In double the same degrees are worth 3
  to 16 bits ON THE SiLU (`degree_plan_oracle.txt`: L1 2^-18 -> 2^-34, L30
  2^-15.7 -> 2^-28.9, L31 2^-15.6 -> 2^-28.7), which is where they will show
  once the floor moves.
* **The single-pass softmax has NO per-layer degree to give.** Its budget of
  seven levels is spent as (15, 7) or (31, 3), and (31, 3) at layer 31 --
  where the exp fit at hb 24.3 is 2^-8.5 and looked like the limiter --
  costs 0.8 bits on the layer and 4 bits on the seam. A degree-3 invsqrt is
  too crude for what the walk hands it, and pricing pairs by their fit error
  over `[norm_lo, norm_hi]` cannot see that. The solver stays in the tree
  behind `SoftMaxCalibration::pair_degrees` (`CHEDDAR_CI_SOFTMAX_PAIR=1`),
  off. Exp 31 WITH invsqrt 7 needs `top` 17, and no ring under the anchor
  affords the tower a level (above).
* **Layer 31 on the oracle is the softmax's Jacobian, and only more boots
  move it** -- which the held-out chain then shows.

### Each operator's precision, layer by layer

On the card (`base32_ledger.txt`, the ledger's probes; from layer 1 on each
probe reads the chain's inherited stream error too, so the layer-0 row is the
one that isolates an operator; `CHEDDAR_CI_CLEAN_EACH=1` restarts every layer
from the reference stream and was written but not run -- the card's time went
to the held-out chains):

    layer 0:  attention RMSNorm 2^-10.08 / rms 2^-10.36 (fit alone 2^-20.6),
              seam (the softmax through its Jacobian) 2^-5.64 / -5.72,
              layer 2^-6.33 / -6.40
    layers 1..30 chained: RMSNorm probe 2^-4.6 .. -6.3, seam 2^-3.3 .. -5.9,
              layer 2^-4.8 .. -6.7 (rms 2^-5.3 .. -6.1, flat)
    layer 31: RMSNorm 2^-5.48, seam 2^-2.48 / -4.23, layer 2^-3.78 / -5.34

In double, the approximation ALONE, 600 held-out wikitext-2 prompts
(`llama_sim.py`, `SIM_ONLY`, `sim_op_table.txt`; p50 over prompts of the
32-layer rms; 0 of 600 non-finite in every run):

| what is approximate | degrees | p50 | L30 | L31 |
|---|---|---|---|---|
| everything, SiLU ladder to 63 (today) | | **2^-5.58** | -11.56 | -5.58 |
| everything, SiLU ladder to 127 | | **2^-12.03** | -12.92 | -12.03 |
| SiLU only | 63 / 127 / 255 | 2^-5.58 / -12.42 / -13.00 | | |
| attention RMSNorm only | 15 / 31 | 2^-13.06 / -18.04 | | |
| FFN RMSNorm only | 15 / 31 | 2^-13.80 / -17.99 | | |
| exp only, invsqrts only, both (Cho k, 31 / 63) | | 2^-13.70 each | | |

The chain is SiLU-limited to the third decimal (all 2^-5.58 = SiLU 2^-5.58)
and the whole of it is layer 31 (L30 2^-11.56); 127 there is the 6.5 bits.
Neither norm is within 7 bits of the limiter at any degree, and the softmax's
three fits sit at 2^-13.7 with the Cho iteration on. So the per-layer degree
that matters is ONE: layer 31's SiLU -- and 255 buys 0.6 bits more only
there.

### The held-out chain runs, for the first time on this path

`gen_b1_pop.py` (64 calibration prompts spread over the Gutenberg stream,
one served prompt from the wikitext-2 test set, `b1_ids.py`; `NITER=2`,
31 / 63, per-layer k) had never met the fused module-basis layer: its Cho
iteration boots the score ciphertexts BETWEEN passes with a native full
`Boot` on `ci16_35`, and the fused setup skipped that ring's native tables
because the oracle single pass never calls it -- `No BootContext available
for num slots: 65536` at layer 0. `CiModelTest.cpp` builds them when the
calibration carries `softmax_niter > 0` (~6 GiB). Two more things the
generator needed: `KMAX=6` (layer 31's m_eff is **134.9** on this split, and
k = 5 left its first window at 1231x against the derived 758; k = 6 puts it
at 40x), and the row-shift margin (below).

| chain (held-out, 3632 boots) | L0 | L1 | L18 | L30 | L31 | worst | median | s |
|---|---|---|---|---|---|---|---|---|
| `pop6` (KMAX 6, RS_MIN 0.10) | 2^-6.01 / -6.26 | -5.52 / -5.77 | -2.97 / -4.81 | -5.29 / -5.05 | **-5.10 / -5.12** | L19 2^-2.77 | 2^-4.74 | 349 |
| `popsink6` (+ the SiLU sink plan) | -6.03 / -6.26 | -5.51 / -5.78 | -2.99 / -4.81 | -5.32 / -5.09 | -4.78 / -5.16 | L19 2^-2.78 | 2^-4.78 | 420 |
| `pop7` (RS_MIN 0.25) | -6.01 / -6.26 | -5.52 / -5.77 | -2.98 / -4.79 | -5.29 / -5.05 | -5.12 / -5.17 | L19 2^-2.77 | 2^-4.74 | 350 |

The Cho counts the generator chose, per layer: k = 4 at L0, L14, L18, L29,
k = 6 at L31, k = 2 at L3, k = 3 everywhere else -- 3632 boots against the
oracle's 3072 (+18 %). That IS "many passes at the last layer, few before".

* **Layer 31 is no longer the worst layer**: 2^-5.10 held-out against
  2^-3.78 on the oracle. Its per-layer lever was k (5 -> 6), one more Boot
  per score ciphertext -- not a degree.
* The worst layers are now 17-24 (2^-2.8 .. -3.8 relative, rms 2^-4.8
  throughout), and the ledger's lane breakdown says where: ONE head per
  layer (L14 lane 25 at 2^-2.58 against the best lane's 2^-5.43, a 2.9-bit
  spread; L17 and L19 lane 31), the seam's rms staying at 2^-4.4 .. -5.0. A
  served (head, row) has left a window calibrated on 64 prompts -- the
  first-invsqrt window and the row shift are the statistical links -- and
  the row-shift floor does nothing (`pop7` = `pop6` to the third decimal,
  because the measured leave-one-out escape already exceeded 0.25 there).
  `calib_audit.py`'s L18 / L31 flag (the exp reading 0.05-0.11 below its
  lower edge) is the same fact seen from the fit. The lever is the
  calibration POPULATION: 64 prompts here against the 384 the 600-article
  protocol used; not run (the generator is ~1 h of host CPU at 384).
* These excursions do not reach the output: the chain contracts them (a
  layer's gain on an input perturbation is 0.73 .. 0.92 in float64), so the
  32-layer closing number is L31's 2^-5.10 / rms 2^-5.12 -- 1.5 bits under
  [SYLPH]'s 2^-6.64 bar at that point, and no perplexity has been run on
  this output (`sylph_precision/sylph_ppl.py` would translate it).
* The sink plan is exact again (sink rows 2^-8.4 .. -10.4 against pop6's
  -7.6 .. -9.2; user rows unchanged) and narrows L1's range 18.3 -> 6.1
  (degree 127 -> 31, a level) and L29/L30 17.4 / 21.9 -> 12.1 / 13.2.
* 32 layers in 349 s with the ledger decrypting every stage: 10.9 s a layer.

### Answer, in one place

Per-layer degrees are the right approach and the tree does it; what it
needed was the caps written as budgets. Where it pays: the SiLU (127 at the
three layers that ask, a level over the old cap, 6.5 bits on held-out
prompts in double; 255 is one slack level away and runs). Where it cannot
pay, and why:

| wanted | stopped by | the number |
|---|---|---|
| SiLU 255 at slack 9 | the FFN turn: 8 levels into 7 | slack 10 gives it (StC 3, coefficients at 1): runs |
| invsqrt 31 in either norm | the weight multiply's pending rescale above StC | slack 10 gives it; buys nothing measurable (norms at 2^-13 already) |
| exp 31 in the single pass | `top` 16 is exact; funding it from the invsqrt (3) loses 4 bits on the seam | a tower ring with top 17 = 42 bits over the anchor |
| a better layer 31 by degree | it is the softmax Jacobian x boot noise | k 5 -> 6: 2^-3.78 -> 2^-5.10 held-out |
| a better held-out middle (L17-24) | one head per layer leaves a window set on 64 prompts | NCAL 384 (the protocol's), not a degree, not RS_MIN |
| the 2^42 pools' 20 / 19 / 18 bits | same-prime crossing; top 16 / 17 (k16 13, k64 11); K 32 / 64 from the wrap-around | +126 / +324 bits of logQ; K is the secret's norm, not the scale |

---

## The 2^35 family (2026-09-11): three presets, a stationary band, any landing

The user's ask, verbatim: reduce `ci16_35` to THREE presets like the 2^42
family -- a flat EvalMod, free landing, the best precision the scale affords
-- and delete the rest. Done on the B200, branch `llama3_CI_Sylph`.

### What the three are

`reference/scripts/ci35_family.py` splices; it does not synthesise. ci16_35's
compute region (q0 = two 25-bit terminals, then `[+2m, -1t] x5, [-3m, +5t]` on
30-bit mains) is kept byte for byte to each pool's encryption level, with its
five terminals, its four 30.02-bit CtS mains and its twelve aux primes -- the
keyless crossing's whole condition, the `_boot` companion trio's, and why the
B = 1 calibration carries over untouched. Only the band is rebuilt: every
level a pair mined to 2^58 to microbits and ORDERED so the signed residues
cancel under EvalMod's doubling (`ci20_family.stationary_pairs`), so the
landing scale is 2^58.000 for every landing and every band length.

    pool             K   dec  band  CtS (bottom -> top)         role                          logQP
    ci16_35_k16_w58  16  19    8    main x2, ter x2 [60,60,50,50] the base ring = ci16_35's shape  1771.5
    ci16_35_k32_w58  32  19    9    ter x2 [50,50]                 the FFN pool (cut at 13 today)  1709.4
    ci16_35_k64_w58  64  17   10    main x2, ter [60,60,50]        the tower ring                  1767.3

All under the 1771.06 anchor, dnum 4, `param_audit --strict` 0 violations 0
warnings, wander +0.00. `log_message_ratio` 5 and `num_double_angle` are
stated in the file. The scale is pinned at 35 twice over: the cycle gives
`m = 30, t = 25` and tops out at 35.23 under `HOIST_MAIN_CAP`, and the tower's
17 + 10 + 3 levels are over the anchor at scale 36 whatever the CtS.

**The rule had to grow.** `LandingLadder`'s CtS rule (terminal triples, then
unused-main pairs, a terminal pair only on the last level) was the 2^42
family's. ci16_35's `[-3m, +5t]` steps put ALL FIVE terminals in the modulus
at levels 3, 9 and 15, so a cut above them must bring the terminal count back
to five, which a triple-first greedy that leaves one over cannot; and its
terminals are 25 bits, so a terminal PAIR is 2^50 -- a transform level, where
the 2^42 family's 23.5-bit pairs are starved. `LandingLadder::CtSPlan`
(mirrored in `landing_ladder.py`) keeps the order and adds a lookahead, with a
terminal pair admitted wherever it is >= 2^49. The 2^42 pools are untouched:
52 device ladders identical to the mirror, 0 plan differences at every
landing. Supported (clean) landings: k16 {2,3,4,5,8,9,10,11,16}, k32
{2,3,4,5,8,9,10,11,14,15,16}, k64 {2,3,4,5,8,9,10,14}; the rest are the exact
ladder above plus slack. k32 cut at HalfBoot 13 is the retired `land13c2e9`
to the level: climb 24, 39 primes, CtS [50.14, 50.19].

### What was measured (B200, `reference/vessl/family_accept.sh`)

| gate | k16_w58 | k32_w58 | k64_w58 |
|---|---|---|---|
| `param_robust_test --strict` | 7/7 | 7/7 | 7/7 |
| Boot residual / landing-scale miss | 2^-15.09 / 0.05 ppm | 2^-14.70 / -0.05 ppm | 2^-14.55 / -0.01 ppm |
| `Bootstrap` SNR | 3.76e10 | 2.75e10 | 4.99e9 |
| `BootstrapPrecisionAgainstSylph` p | 14.90 | 15.00 | 14.47 |
| crossing pool -> ladder at HalfBoot 13 (rel. residual) | 2^-14.89 | 2^-14.99 | 2^-14.60 |
| boot at the pool's own landing (median of 10) | 25.0 ms | 74.0 ms | 27.1 ms |

`ci16_35` measured p = 15.05 on the A100; the run-to-run spread is 0.07-0.20.
The ratio curve (`CHEDDAR_BOOT_MSG_RATIO` 3/4/5/6): k16 14.97 / 15.33 / 14.72 /
14.82, k64 14.85 / 14.58 / 14.71 / 13.65 -- flat within half a bit until 6
on K 64, so the pools ship 5, the value every fixture always passed. Band 60
(30-bit pairs) measures 2^-15.15 at the crossing against band 58's 2^-15.17 on
the same pool, and puts K16 and K64 sixteen bits over the anchor: **at scale
35 the precision is the additive floor's, ~15 bits, whatever the band or the
ratio** (`CI_PARAM_20BIT.md` 1) -- "최대한" is what the family already is,
and more needs more scale, which the leg cannot afford (the 2^42 verdict
above). The one surprise is the K 32 pool's native boot at 74 ms: two CtS
levels make each transform radix-256. The layer never runs a native boot on
that ring (its crossings are `HalfBootModule`, whose CtS is the module
basis), and the short climb is what it pays for: **k32 cut at 13 by the
ladder 64.5 ms against 72.5 ms by slack 6 (0.89x)**.

### What changed in the tree

Deleted: `ci16_35.json` and sixteen `ci16_35_land*`/`stc2` files, and
`gen_landing.py` (the v3 solve is what the stationary band makes unnecessary).
`ci16_35_k16_w58` is the base ring everywhere `ci16_35.json` was (29 tests and
scripts renamed; gtest instances are `*ci16_35_k16_w58_json*` now). The StC-2
shapes (`stc2`, `land17c4e8s2`, the channel ring `land11c4e8s2`) have no
equivalent: a pool's StC count is fixed and slack only lowers a landing; their
results stand in Doing.md 7.36-7.39. `ci_model_test` builds its rings through
`MakeRing`: `CHEDDAR_CI_{BASE,FFN,LEG}_LANDING` name the HalfBoot landing the
pool is cut to (unset = the file as it is), and `sylph_run2.sh` defaults to
`ci16_35_k32_w58.json` at 13 and `ci16_35_k64_w58.json` at its own 17.
`boot_landing_test` builds its landing ring from the k32 pool by the same cut;
`ci_sinc_basis_test` runs the tower basis on the k64 pool; `landing_ladder_test`
covers all six pools.

### The B = 1 / T = 128 chains on the family (item 2)

Both 32-layer chains, seed 11, the same calibrations as the runs above
(`reference/vessl/family_layer.sh`; ledgers in
`reference/audit/2026-09-11_b200_degrees/fam_*_ledger.txt`):

| chain | before (ci16_35 + land13c2e9 + land17c3e10) | on the family (k16 + k32 cut at 13 + k64) |
|---|---|---|
| oracle, L31 rel / rms | 2^-3.78 / 2^-5.34 | **2^-3.75 / 2^-5.22** |
| oracle, layers 0..30 | -- | every layer within 0.1 bit of before (L0 2^-6.33 = 2^-6.33) |
| held-out (pop6), median rel | 2^-4.81 | **2^-4.87** |
| held-out, worst | L19 2^-2.77 | L19 2^-2.92 |
| held-out, L31 rel / rms | 2^-5.10 / 2^-5.12 | 2^-5.03 / 2^-5.12 |
| held-out, 32 layers | 349 s | 395 s (11.5 s a layer, ledger on) |

The family is the same layer to the run-to-run spread, which is what a
byte-identical compute prefix and a band stationary to 2e-3 bits should give.
The FFN ring's cut (`[landing ladder] ci16_35_k32_w58.json for
CHEDDAR_CI_FFN_LANDING=13: landing 10 ... climb 24, 39 primes, CtS [50.14,
50.19]`) is printed by the layer itself.

**A calibration-reading defect found on the way.** `gen_b1_pop.py` widens
every row shift by its leave-one-out margin `a` (0.18 .. 0.54 across the 32
layers) and walks its windows with `span_raw = (1 + a)(s_max - s_min)`, which
it stores as `span` (m_eff) -- `s_raw_min/max` stay the population's raw
extremes. `CiModelTest` (and `CiBatchTest`) rebuilt `span_raw` from the raw
extremes, so the exp's `u = 1 + 2 (S - shift) / span` reached -1.36 .. -2.09
for the lowest live scores of every row of the held-out chain, outside the
[-1, 1] the exp is fitted on (a degree-15 interpolant of `exp(hb (u - 1))`
returns ~0.05 there instead of ~1e-4). The exponent itself was right --
`m_eff` shrank by the same factor -- so the walk was the generator's up to the
polynomial's extrapolation. Both tests now take the span from `span` when the
bundle carries it (a no-op for the oracle and for `gen512.py`, whose `span` is
exactly the raw difference: ratio 1.000000 on every layer).

**Measured (`fam_pop6_fix`, same seed):** the rms column does not move (every
layer within 0.05 bits), and the max-relative column moves by +0.01 .. +0.1
at layers 0-15 and +0.2 .. +0.36 at layers 16-30, a mean of +0.12 bits the
WRONG way (median 2^-4.78 against 2^-4.87, worst L19 2^-2.73 against
2^-2.92, L31 2^-5.06 against 2^-5.03). The mechanism is a level, not the
polynomial: the exponent `hb (u - 1)` was already the generator's on both
readings (`m_eff` shrank by the same factor `span` grew), so the walk was
the same up to the extrapolation -- but the exp's DEGREE is chosen from `hb`
by the 2^-16 rule, and the widened `hb` (x 1.18 .. 1.54) takes it from 7 to 9
at 19 of the 32 layers (1, 2, 5, 6, 8, 9, 11, 15-17, 19, 21, 23, 25-28, 30),
one more level in the exp before the first invsqrt, and what the chain then
accumulates over layers 16-30 is a fifth of a bit on the worst element. The
fix stays: the certified upper end of the first window
(`softmax_first_hi_certified`) is a statement about the walk the generator
did, and a polynomial read outside the interval it was fitted on is the one
thing this project's calibration rule forbids -- but the number to quote for
the held-out chain on the family is the fixed one, 2^-4.78 median.

**The FFN pool at its own 19 (`CHEDDAR_CI_FFN_LANDING=19 CHEDDAR_CI_FFN_SLACK=15`,
`fam_pop6_f19`).** Six more levels for the slot-domain half: both RMSNorms go
from degree 15 to 31 wherever their window asks (most layers), layer 31's
SiLU from 127 to 255, the FFN ring climbs to 30 (46 limbs) instead of 24 (39).
The held-out chain: median 2^-4.78 against 2^-4.87 at 13, worst L19 2^-2.71
against 2^-2.92, L31 2^-4.84 against 2^-5.03 -- the same to the spread, and
11.5 s a layer either way. So on the card the FFN half is not level-limited,
which is what the single-layer A/Bs of the section above already said of the
SiLU (127 invisible at the crypto floor); the cut at 13 stays the default.

## Wiring

**Slim is wired**, into the place section 3.4 names: `CiSinCAttention`'s
auxiliary track, through `SoftMaxCalibration::slim_j`. Default 0, so the
shipped walk is untouched operation for operation. Three things came out of
doing it rather than describing it:

- **The slim period is derivable, so it is derived.** The reduction tree
  rotates by `stride << t` for `t = 0..3` with `stride = num_slots / rank`, so
  after it slot `s` and slot `s + stride` hold the same row's norm.
  `PrepareSoftMax` computes the period from the layout and refuses a `j` it
  cannot hold.
- **It needs no new rotation key.** Algorithm 1's fold rotates by
  `block * 2^(l-1)` for `l = 1..j`, and the reduction tree already asks for
  exactly that set for every `j <= 4` — which is every `j` the period admits.
  There is an assert saying so.
- **Appendix D's fold is implemented, and it is out of reach HERE — which is
  the most useful thing doing the wiring produced.** The fold makes Algorithm 1
  cost `k` levels for degree `2^k`, exactly Paterson-Stockmeyer's cost for
  `2^k - 1`, and it needs `j == k`. But how slim a ciphertext is decides how
  many blocks the tree gets, and this auxiliary track is barely slim: the norm
  is broadcast over the row's whole period, `num_slots / rank = 4096`, so only
  16 blocks fit and `j <= 4`. `j == k <= 4` means degree at most 16, and
  degree 16 on `[1/live, 1]` is 2^-1.2. So what is actually available is
  `j = 4, k = 6`: **degree 64 in 7 levels where Paterson-Stockmeyer buys 63 in
  6 — one extra level for half the multiplications**, not a free win.

  The payoff is gated on **compacting the auxiliary track to one slot per
  row**, which is exactly what §2.3 assumes when it says that track "operates
  on only `d` values" and is therefore "performed on sparsely-packed
  ciphertexts". At a period of 128 the tree would get 512 blocks, `j <= 9`,
  and the fold would reach degree 512. That compaction, not appendix D, is the
  prerequisite — and `SoftMax.h` had already flagged it ("here it is broadcast
  across every slot by the reduction and bootstrapped at full width").

**Eq. (5) is not wired.** `SylphPcmm` is built and host-tested, and so is the
`tau^2` it needs (`TauPermutation`, one `SlotPermute` at one level and 64
diagonals for `d = 128` -- measured, and cheaper than the 255 the square
transpose costs, because an even power halves the orbit of `n j mod d`). What
is missing is not machinery but a decision that is genuinely open: the T = 4096 branch's `CiPcAttention`
  reaches PC-attention through [KANG] Algorithm 1, which is **depth 1 with no
  rotation, no automorphism key and no relinearization key at all**. Eq. (5)
  is depth 1 with `O(sqrt d)` rotations. So the paper's algorithm is not
  obviously the better one *here*, and the comparison is the point — which is
  why both should exist rather than one replacing the other.

## Running it

No GPU:

    cmake --build build --target slim_poly_test sylph_pcmm_test
    ./slim_poly_test  --gtest_filter='SlimMath.*'
    ./sylph_pcmm_test --gtest_filter='SylphPcmmMath.*'

With one GPU:

    ./slim_poly_test  --gtest_filter='*SlimAlgorithmOne*'
    ./sylph_pcmm_test --gtest_filter='*EquationFiveOnEncrypted*'

Then the existing regression, unchanged by any of this:

    ./softmax_test && ./ci_model_test   # per CLAUDE.md section 5

Host-side calibration (CPU only, needs `llama3_all` and `wiki_ids.npy`):

    python3 reference/scripts/sylph_tables.py <model> <ids> 32 none,sink,resid,vo
    QUAROT_VO=1 python3 reference/scripts/quarot_export.py <model> <out>

The knobs the card runs used (all default OFF; `ci_model_test` prints a line
at the first layer whenever one is read, so a log proves it):

    CHEDDAR_RNG_SEED=11          a fixed draw -- A/B pairs share it
    CHEDDAR_CI_SILU_DEG=127      force the SiLU degree (0 = SiLuDegree's rule)
    LLAMA3_REF_DIR=<ref_band>    a calib.json carrying `silu_bands`
                                 (silu_band_inject.py <calib> <out> 8 all);
                                 CHEDDAR_CI_SILU_BANDS=0 ignores it
    LLAMA3_REF_DIR=<ref_sink>    `silu_sink` + silu_restore_L<NN>.f64
                                 (silu_sink_inject.py <all> <ref> <out>);
                                 CHEDDAR_CI_SILU_SINK=0 ignores it
    CHEDDAR_CI_FFN_SLACK=10      the FFN ring's slack (default 9): StC one
                                 lower, one more level for the SiLU / invsqrt
    CHEDDAR_CI_SILU_MAX_DEG=63   cap the SiLU ladder below its budget (the
    CHEDDAR_CI_RMS_MAX_DEG=15    old literals, for an A/B); 0 = the budget
    CHEDDAR_CI_EXP_DEG=15        pin the single pass's exp degree (invsqrt 7)
    CHEDDAR_CI_SOFTMAX_PAIR=1    the (exp, invsqrt) pair solver -- loses
    CHEDDAR_CI_CLEAN_EACH=1      every layer from the reference's clean
                                 stream, one process (each layer's OWN ledger)
    LLAMA3_REF_DIR=<pop> LLAMA3_INPUT=<pop>/input_pop.f32
                                 a held-out bundle (gen_b1_pop.py, KMAX=6
                                 RS_MIN=0.25 WIKI_IDS=b1_ids.npy); the test
                                 builds ci16_35's native boot for its Cho pass
    CHEDDAR_CI_FFN_PARAM=ci16_35_k32_w58.json CHEDDAR_CI_FFN_LANDING=13
    CHEDDAR_CI_LEG_PARAM=ci16_35_k64_w58.json CHEDDAR_CI_LEG_LANDING=17
    CHEDDAR_CI_BOOT_PARAM=ci16_35_k16_w58.json CHEDDAR_CI_BASE_LANDING=19
                                 the 2^35 family (2026-09-11): one pool per K,
                                 each cut by LandingLadder to the HalfBoot
                                 landing named (unset = the file as it is);
                                 these are sylph_run2.sh's defaults

The B200 recipe (`reference/vessl/sylph_b200_setup.sh` from a git bundle,
`sylph_b200_data.sh`, `sylph_run2.sh <tag> <first> <layers> [ENV=..]` with
`BUILD=build100b`, the suites `sylph_suite*.sh`, `sylph_host.sh`):
`/workspace` has a QUOTA -- a 13 GB stale module-basis cache had to go before
the 27 GB export fit -- and `pkill -f <pattern>` kills the ssh shell whose
command line contains the pattern (use `[.]` / `[t]` in the regex).

The prefix study on a GPU (cupy; `SIM_GPU=0` is the same code on numpy):

    python3 sylph_prefix_gpu.py <ids.npy> 528 @prefix_search.json
    SINKCHECK=1 python3 sylph_prefix_gpu.py <ids.npy> 16 bos2,bos1,A
    python3 prefix_heldout.py <pp dir> <ids.npy.src.json>
