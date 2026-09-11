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
| **3.4** | **slim polynomial evaluation — lemma 1, eq. (2), eq. (3), Algorithm 1, theorem 1, appendix D** | **HOST** | **`SlimPolyMath.h` + `SlimPoly.h`, new on this branch — see below** |
| 3.5 | end-to-end latency | n/a | a measurement, not an algorithm |

## 3. Section 4 — long heterogeneous prompts

| § | What the paper specifies | Status | Where |
|---|---|---|---|
| 4.1 | public prefill in the clear, private prefill on the encrypted tail | YES | `PublicPrefill`, `CiPcAttention` |
| 4.1 | PC-attention and CC-attention split | YES | `CiPcAttention` / `CiSinCAttention` |
| **4.2** | **eq. (5): depth-one PCMM, `tau^(l+1)(B) -> tau^l(C)`, BSGS, `O(sqrt d)` rotations** | **HOST** | **`SylphPcmmMath.h` + `SylphPcmm.h`, new on this branch** |
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
| `[BOS, 
]` | 0.9136 | 27.79 | 80.88 |
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

### What was implemented, and the one thing that was not

`gen_b1_pop.py` gains `SILU_SINK=1`: it measures `gate_absmax_user`, derives a
public per-token factor `s` that brings the sink rows inside the user interval,
and writes the public correction `SiLU(g) - SiLU(g s)` per (sink token,
channel). The identity `y = (SiLU(g s) + restore) up` is exact at every row,
`s` rides the gate crossing's existing multiply and `restore` is a plaintext
add, so it costs no level. It also CHECKS that the sink rows' gate really is
prompt independent rather than asserting it. Default off.

**The crypto side of that restore is deliberately NOT written.** `s` is a
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

---

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
