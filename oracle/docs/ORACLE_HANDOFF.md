# S2 HANDOFF — the plaintext oracle

Session **S2**, branch `s2-plaintext-oracle`, worktree
`Projects\cheddar-fhe-s2-oracle`. Everything below lives under `oracle/` and
nothing outside it was touched.

This document is the interface. It says what the oracle guarantees, what it
does not, what Sessions 4–7 should feed it and expect back, and what S2 needs
from S0, S1 and S3 before some of its own numbers can be finalised.

---

## 1. The verified Llama-3-8B configuration

Every value was read out of a file in this checkout or in the read-only
`Projects/HEonGPU` tree on 2026-08-14. `llama3_oracle_cli config` prints this
with the file and line for each row.

| field | value | evidence |
|---|---|---|
| `vocab_size` | 128256 | `HEonGPU/benchmark/fetch_llama3_weights.py:60` |
| `hidden_size` (d_model) | 4096 | `fetch_llama3_weights.py:61`; `reference/LLAMA3_8B_LAYER_FLOW.md:987-988` |
| `intermediate_size` | 14336 | `fetch_llama3_weights.py:62`; same flow-doc lines |
| `num_layers` | 32 | `fetch_llama3_weights.py:63` |
| `num_heads` | 32 | `fetch_llama3_weights.py:64` |
| `num_kv_heads` | 8 | `fetch_llama3_weights.py:65` |
| `head_dim` | 128 | `fetch_llama3_weights.py:66` |
| `rope_theta` | 500000.0 | `fetch_llama3_weights.py:67`; `HEonGPU llama3_batch.cuh:477` ("Llama-3's base is 500000; Llama-2's is 10000"), `llama3_batch16.cuh:600`, `llama3_prep.cuh:108`, `llama3_rect.cuh:1175` |
| `rms_norm_eps` | 1e-5 | `fetch_llama3_weights.py:68`; `HEonGPU llama3.cuh:510`, `llama3_batch.cuh:843`, `llama3_batch16.cuh:388`, `llama3_prep.cuh:236`, `llama3_rect.cuh:1007` |
| `max_position_embeddings` | 8192 | `reference/LLAMA3_8B_LAYER_FLOW.md:2442` |

Cross-checks that do not depend on those constants: `4096 == 32 × 128`;
GQA group `32 / 8 = 4`; `kv_channels = 8 × 128 = 1024`, matching
`LLAMA3_8B_LAYER_FLOW.md:3003` ("kv_channels stays at 1024 against q_channels
4096").

**Assumed, not verified here** — flagged so a later session confirms it against
a real `config.json` rather than inheriting it: no bias on any projection;
RMSNorm (not LayerNorm) applied pre-attention and pre-FFN with the residual
added after each sublayer; **no RoPE frequency scaling** (Llama 3.0 has none;
Llama 3.1 adds a `rope_scaling` block and this oracle is *wrong* for 3.1 until
that is added); SwiGLU FFN.

### The one convention that is not a constant

**RoPE channel pairing.** `half_split` (HuggingFace `rotate_half`: channel `c`
pairs with `c + head_dim/2`) is the default and is correct for HF-format
safetensors. `interleaved` (channel `2c` with `2c+1`) is correct for an
original Meta `consolidated.00.pth` export. Both are "correct Llama-3", both
are orthogonal, and both produce identically-shaped, identically-normed
tensors — **no shape check and no norm check distinguishes them.** The unit
test `Rope.ConventionsDiffer` exists to stop the two silently collapsing.
Whoever loads real weights must settle this once against a trusted forward pass
and record the answer.

---

## 2. Numerical conventions the oracle commits to

1. **float64 everywhere.** No float32 or bfloat16 pass. The oracle's own
   rounding is ~12 orders of magnitude below the CKKS error it measures
   (Cheddar's measured bootstrap error is 4.46e-05 average absolute). Comparing
   the oracle against a float32 HuggingFace run will show ~1e-7 differences
   that belong to the float32 run.
2. **Every reduction is a single sequential float64 accumulator in ascending
   index order.** No pairwise summation, no Kahan, no BLAS, no threading, no
   reassociation. `/fp:precise` on MSVC, `-ffp-contract=off -fno-fast-math` on
   GCC/Clang — FMA contraction is off because it changes the result and is not
   portable between hosts.
3. **Randomness is SplitMix64 plus an explicit Box–Muller**, written inside the
   project. Nothing depends on `std::normal_distribution`, whose variate
   consumption is implementation-defined; the same seed gives the same tensors
   on MSVC and on Sicily's GCC 10.
4. **The causal mask multiplies the exponentials by 0/1**, it does not add
   −∞ to the scores, because that is what an encrypted SoftMax does. Masked
   probabilities are therefore exactly `+0.0`.
5. **RMSNorm computes `1/sqrt(sum_c x² / hidden + eps)`** with eps *inside* the
   square root. `sum_sq` and `mean_sq` are both exported so a folded-mean
   encrypted variant can be calibrated against the interval it actually sees.
6. **RoPE is applied to Q and K only**, angle for pair index `c` is
   `position × theta^(-2c/head_dim)`, position absolute.
7. **Scores are scaled by `1/sqrt(head_dim)` before the SoftMax shift**, so
   every exported score is already scaled.
8. **Determinism, stated precisely.** Two runs of the same binary on the same
   host with the same inputs are bit-identical and the exported checksums prove
   it. **Across hosts and compilers** the results agree only to the last ulp of
   libm's `exp`/`log`/`cos`/`sin`/`pow`, which is *not* guaranteed bit-identical.
   **Cross-host equality must be asserted with the tolerance-based comparison
   utility, never with checksums.**

---

## 3. The boundary contract — 29 tensors per layer

`llama3_oracle_cli boundaries` prints this list with descriptions. Names are
prefixed with the layer, e.g. `L0.attn.probs`. Layout conventions: activations
are `[tokens, channels]` row-major (channel fastest); weight matrices are
`[in_features, out_features]` — **the transpose of HuggingFace's `[out, in]`**;
score-shaped tensors are `[query_head, query_token, key_token]`.

**None of this is a ciphertext packing.** S0 owns the mapping from these
logical shapes onto slots; the oracle fixes only values.

| # | name | axes | shape | fit role | produced by |
|---|---|---|---|---|---|
| 1 | `x_in` | token,channel | T × 4096 | — | previous block |
| 2 | `attn_norm.sum_sq` | token | T | rsqrt (folded variant) | RMSNorm reduction |
| 3 | `attn_norm.mean_sq` | token | T | **rsqrt** | RMSNorm |
| 4 | `attn_norm.inv_rms` | token | T | — | 1/sqrt fit output |
| 5 | `attn_norm.out` | token,channel | T × 4096 | — | RMSNorm |
| 6 | `q_proj` | token,channel | T × 4096 | — | Q projection |
| 7 | `k_proj` | token,channel | T × 1024 | — | K projection |
| 8 | `v_proj` | token,channel | T × 1024 | — | V projection |
| 9 | `q_rope` | token,channel | T × 4096 | — | RoPE |
| 10 | `k_rope` | token,channel | T × 1024 | — | RoPE |
| 11 | `attn.scores` | head,query,key | 32 × T × T | — | QKᵀ, scaled |
| 12 | `attn.exp_input` | head,query,key | 32 × T × T | **exp** | score − shift |
| 13 | `attn.exp` | head,query,key | 32 × T × T | — | exp, mask applied |
| 14 | `attn.denominator` | head,query | 32 × T | **reciprocal** | row sum |
| 15 | `attn.reciprocal` | head,query | 32 × T | — | 1/x fit output |
| 16 | `attn.probs` | head,query,key | 32 × T × T | — | normalise |
| 17 | `attn.context` | token,channel | T × 4096 | — | PV |
| 18 | `o_proj` | token,channel | T × 4096 | — | O projection |
| 19 | `x_mid` | token,channel | T × 4096 | — | residual add |
| 20 | `ffn_norm.sum_sq` | token | T | rsqrt (folded) | RMSNorm reduction |
| 21 | `ffn_norm.mean_sq` | token | T | **rsqrt** | RMSNorm |
| 22 | `ffn_norm.inv_rms` | token | T | — | 1/sqrt fit output |
| 23 | `ffn_norm.out` | token,channel | T × 4096 | — | RMSNorm |
| 24 | `ffn.gate` | token,channel | T × 14336 | **silu** | gate projection |
| 25 | `ffn.up` | token,channel | T × 14336 | — | up projection |
| 26 | `ffn.silu` | token,channel | T × 14336 | — | SiLU fit output |
| 27 | `ffn.swiglu` | token,channel | T × 14336 | — | gate product |
| 28 | `ffn.down` | token,channel | T × 4096 | — | down projection |
| 29 | `x_out` | token,channel | T × 4096 | — | residual add |

Two of these are easy to get wrong and are worth reading twice.

* **`attn.exp_input` includes causally masked positions on purpose.** The
  encrypted circuit masks `exp(u)`, not `u`, so `exp` is evaluated at every
  position including the masked ones and the fit interval must cover them.
  Under row-max shifting an *unmasked* `u` is ≤ 0 but a *masked* `u` can be
  positive. A fit interval measured over unmasked positions only is too narrow,
  and the resulting failure looks like Chebyshev instability rather than like a
  range error.
* **`attn.denominator` is where calibration pays.** See §6.

### Serialization

One tensor per file, magic `CHDORC1`, version 1, all integers little-endian,
no padding: 8-byte magic, `uint32` version / dtype / rank / flags,
`uint64 dims[rank]`, length-prefixed name, length-prefixed JSON metadata,
`uint64 payload_bytes`, payload, `uint64` FNV-1a checksum of the payload. A
directory also carries `manifest.json` with the configuration, the weight
source, the numerical conventions, and one record per tensor (shape, axis
names, metadata, checksum, min/max/mean/rms).

`llama3_oracle_cli conventions` prints the full spec **and a copy-paste Python
reader** (numpy only, ~15 lines). `ReadTensorFile` is the C++ reader; it
verifies magic, version, declared payload size and checksum, and names which
of those failed.

---

## 4. Acceptance thresholds

`llama3_oracle_cli thresholds` prints these with full rationale. Every
criterion is **relative**; `max_abs` is reported for diagnosis and is never a
pass condition. (HEonGPU saw a SwiGLU read as broken at 6.9e-01 absolute and
ordinary at 1.8e-02 relative — a fixed absolute bound fails a correct circuit
the moment the model width moves.)

The primary metric is `rel_l2 = ‖test − ref‖₂ / ‖ref‖₂`. `gain = ⟨t,r⟩/⟨r,r⟩`
is checked separately because a systematic scale drift — a mismanaged rescale,
a missing factor — shows there while `rel_l2` is still small. `cosine` is
checked because a near-zero reference makes `rel_l2` uninformative.

| tier | rel_l2 ≤ | pointwise rel ≤ | cosine ≥ | \|gain−1\| ≤ |
|---|---|---|---|---|
| primitive | 1e-5 | 1e-4 | 1−1e-9 | 1e-5 |
| module | 1e-2 | 5e-2 | 1−1e-4 | 1e-2 |
| block | 2e-2 | 1e-1 | 1−5e-4 | 2e-2 |
| multi-block (n) | 2e-2·√n | 2e-1 | 1−2e-3 | 5e-2 |

Per-module tightenings (`ThresholdForModule`):

| module | rel_l2 ≤ | note |
|---|---|---|
| `rope`, `linear`, `projection`, `residual`, `mask` | **1e-5** | **no polynomial in the path — held to the primitive tier.** A fit-sized tolerance on a rotation by plaintext cos/sin, a plaintext matrix product, an addition, or a 0/1 multiply would hide a real defect |
| `rmsnorm` | 5e-3 | 1/sqrt's branch point at zero sets its error; at d_model 4096 the summed square concentrates and the interval collapses towards 1.1:1 |
| `softmax` | 1e-2 | two fits in series with a mask between them |
| `silu`, `swiglu`, `ffn` | 3e-2 | the fit interval tracks the model width; quote relative error only |
| `attention` | 1.5e-2 | whole sublayer |

Two criteria that are deliberately **not** relative:

* **MaskedResidue.** A masked probability is exactly `+0.0` in the oracle, so
  it has no relative error. Test it absolutely against the unmasked peak:
  `|p_masked| ≤ 1e-5 × max_unmasked_p`. HEonGPU measured 8.39e-09 for a
  masked-off key after calibration and 5.61e-07 before — the number moves with
  the calibration, which is exactly why it must be its own criterion rather
  than folded into `rel_l2` over the whole score tensor, where 99% of the mass
  would drown it.
* **ProbabilityMass.** Each attention row should sum to 1: `|Σp − 1| ≤ 1e-2` at
  module tier. This catches a wrong reciprocal interval, which `rel_l2` on the
  output can partially absorb.

**Status of these numbers.** The primitive tier is anchored on Cheddar's own
two measured numbers (the bootstrap error) and is otherwise a **hypothesis** —
no per-operation error trace for Cheddar exists yet, and **S3 is producing
exactly that.** When it lands, `ThresholdPrimitive()` must be re-anchored on
the measured per-op error at the level the module actually runs at. The module,
block and multi-block tiers are anchored on HEonGPU measurements at the real 8B
widths and transfer as **orders of magnitude, not digits**: HEonGPU ran 40–60
bit primes, Cheddar's presets are ~30-bit with 32-bit words. The multi-block
√n composition exponent is **unmeasured** — the residual stream is a shared
path, so a correlated error would grow like `n` instead. Treat a tier-4 pass as
weak evidence until the growth is measured.

---

## 5. Test vectors

`llama3_oracle_cli vectors` prints the catalogue; `llama3_oracle_cli selftest`
runs every one and checks the structural invariants. Seven of the thirteen are
deliberate controls.

| vector | stresses | control |
|---|---|---|
| `baseline` | the reference point the others are read against | |
| `single_token` | the degenerate causal row; denominator exactly 1 | |
| `two_tokens` | the smallest non-trivial causal structure | |
| `mask_boundary` | diagonal, first row, last row, upper triangle | ✔ |
| `near_zero_norm` | the **lower** bound of the 1/sqrt interval | ✔ |
| `zero_row` | `mean_sq` exactly equal to `rms_norm_eps` | ✔ |
| `repeated_tokens` | catches a RoPE that was wired in but never applied | ✔ |
| `repeated_tokens_no_rope` | closed form: `p = 1/(q+1)` exactly, denominator walks the **full worst-case `[1, T]`** | ✔ |
| `large_scores` | the **upper** bound of the exp interval | ✔ |
| `extreme_magnitudes` | six decades of dynamic range through the projections | ✔ |
| `outlier_channels` | a residual stream with attention-sink-shaped outliers | |
| `unscaled_weights` | reproduces the documented HEonGPU failure; **expected to be unfittable** | ✔ |
| `long_window` | the reciprocal interval widens with the row length | |

Invariants checked on every vector (`CheckInvariants`): attention rows sum to
1; masked probabilities are exactly zero; row 0 is one-hot on key 0; every
recorded tensor is finite; probabilities lie in [0,1].

---

## 6. Calibration — measured ranges and degree recommendations

See §7 for the measured table. Read these caveats first; they decide whether a
number may be quoted.

1. An interval measured on N token windows is a **lower bound** on what a
   deployment sees. Widen it deliberately and say by how much.
2. With synthetic weights every range is a property of the **synthetic
   distribution**, not of Llama-3. The weight scaling (`1/sqrt(fan_in)`) was
   chosen so a unit-variance input gives unit-variance scores, which puts the
   synthetic score distribution in the same decade a trained model produces.
   That is a modelling choice, and it is the entire reason these ranges are
   plausible rather than the e⁷⁵ dynamic range HEonGPU measured from unscaled
   random weights. The `unscaled_weights` vector reproduces the bad regime on
   purpose.
3. Minima and maxima are exact; quantiles come from a bounded reservoir and are
   approximate. **Build intervals from the extremes**, read quantiles only to
   see how much of the interval is tail — a Chebyshev fit does not degrade
   gracefully outside its interval, it diverges.
4. Every interval must be re-measured when the sequence length, the SoftMax
   shift, the score scale or the weights change. All four move it.

### Which error a degree is judged on

* `rsqrt`, `reciprocal` → **pointwise relative**. Their output is a scale
  factor (`1/sqrt` multiplies the normalised row, `1/denominator` multiplies
  the attention numerator), so a relative fit error becomes a relative answer
  error at every magnitude, small values included.
* `exp`, `silu` → **relative to peak** (`max|err| / max|f|`). Their output is
  summed or weighted rather than used as a scale: `exp` feeds a normalising sum
  in which small terms carry almost no mass, and SiLU feeds a 14336-term
  contraction. Judging those pointwise reports an error the circuit does not
  suffer, and for SiLU it is meaningless outright — SiLU crosses zero, so the
  pointwise ratio near the root measures where the root is, not how good the
  fit is.

Both columns are printed for every row so the choice can be argued with.

### A degree sweep can fail in two different ways

* *"No candidate degree reached the target"* is an **approximation** failure.
  Narrow the interval, split the domain, or accept a larger error. Raising the
  degree also works and costs levels.
* *"REJECTED ON DYNAMIC RANGE"* is **not an approximation failure at all**. The
  function's own values span more decades over the interval than the ciphertext
  scale carries, so the result is unrepresentable however well it is
  approximated and **no degree helps**. This is HEonGPU's documented failure:
  an exp over a span of e⁷⁵ did not fail loudly, it returned values that
  outgrew int64 at decrypt and looked like a library fault. The 1e12 budget in
  `RecommendDegree` is a **placeholder** until Cheddar's usable dynamic range
  is measured on hardware.

---

## 7. Measured results (synthetic weights, real 8B shape)

Full reports: `oracle/reports/synthetic_8b_t128/` and
`.../synthetic_8b_t512/` (`calibration.txt`, `fits.txt`, `calibration.json`).
Reproduce with:

```
llama3_oracle_cli calibrate --tokens=128 --layers=1 --out=<dir>
```

Configuration: hidden 4096, intermediate 14336, 32 heads over 8 KV heads,
head_dim 128, 1 layer, 128 tokens, `rms_norm_eps` 1e-5, `rope_theta` 500000,
RoPE `half_split`, seed 20260814, **synthetic weights scaled `1/sqrt(fan_in)`**.
Runtime 42 s on one CPU core, ~1.75 GiB resident.

**Every number here is a property of the synthetic weight distribution, not of
Llama-3.** They are the right order of magnitude and the right *shape* of
problem; they are not a substitute for a measurement on real weights.

### 7.1 Measured input ranges

| function | channel | observed range | with 5% margin | dynamic range |
|---|---|---|---|---|
| `1/sqrt` | `attn_norm.mean_sq` ∪ `ffn_norm.mean_sq` | [0.9499, 2.0217] | [0.8963, 2.0753] | **1.52** |
| `exp` (row-max shift) | `attn.exp_input` | [−7.981, +5.266] | [−8.644, +5.929] | 2.13e6 |
| `exp` (fixed shift) | `attn.exp_input` | [−10.028, 0] | [−10.529, +0.501] | 6.17e4 |
| `1/x` (row-max shift) | `attn.denominator` | [1.000, 30.75] | [0.500, 32.24] | **64.5** |
| `1/x` (fixed shift) | `attn.denominator` | [0.001007, 2.654] | [0.000504, 2.787] | **5534** |
| SiLU | `ffn.gate` | [−4.843, +4.843] | [−5.327, +5.327] | (crosses zero) |

The `1/sqrt` interval is the one the record predicted: at d_model 4096 the
summed square concentrates and the interval collapses to a 1.5:1 span. That is
why RMSNorm is the cheapest of the four fits and why a small-width test of the
same circuit is legitimately worse — do not read a small-width RMSNorm error as
a defect.

The `exp` upper bound under a fixed shift is exactly 0 before the margin, by
construction (the shift *is* the measured maximum). Under a row-max shift it is
**+5.266, not 0** — that is the masked positions, and it is the concrete form
of the warning in §3.

### 7.2 Measured degrees

Target: max relative approximation error 1e-4 on a 20001-point float64 grid.
`depth = ceil(log2(deg+1))`.

**Part A — the implementable circuit (one global fixed shift):**

| function | interval | degree | depth | achieved |
|---|---|---|---|---|
| `1/sqrt` | [0.896, 2.075] | **7** | 3 | 1.58e-06 pointwise |
| `exp` | [−10.53, +0.50] | **15** | 4 | 9.18e-09 err/peak |
| `1/x` | [0.000504, 2.787] | **none ≤ 127** | — | 6.40e-02 pointwise at degree 127 |
| SiLU | [−5.33, +5.33] | **15** | 4 | 9.69e-05 err/peak |

**Part B — the unreachable ideal (per-row maximum), i.e. the upper bound on
what a better shift strategy could buy:**

| function | interval | degree | depth | achieved |
|---|---|---|---|---|
| `1/sqrt` | unchanged | 7 | 3 | 1.58e-06 |
| `exp` | [−8.64, +5.93] | 15 | 4 | 2.04e-07 err/peak |
| `1/x` | [0.500, 32.24] | **63** | 6 | 2.20e-07 pointwise |
| SiLU | unchanged | 15 | 4 | 9.69e-05 |

### 7.3 The finding that matters: the SoftMax shift

> A per-row maximum gives a reciprocal interval of ratio **64.5**, which
> degree 63 fits to 2.2e-07. A single global shift gives a ratio of **5534**,
> which **no degree up to 127 fits at all** — degree 127 is still 6.4% wrong.
> A row maximum is not a low-degree polynomial, so CKKS cannot compute one.

This is the largest open design question the oracle surfaces, and it is
**S5's first decision, gated on S0's packing**. The options, in the order they
should be considered:

1. **Shift per row-block** rather than globally, if the packing lets a block of
   rows share one plaintext constant. This narrows the ratio by exactly the
   spread of the per-block maxima and costs only the constants.
2. **Refresh the denominator on a narrow auxiliary track** so the reciprocal is
   evaluated at a fixed depth over a fixed range — the `refresh_denominator`
   idea in the HEonGPU record, which reports the fit moving off the wide track
   entirely.
3. **Split the `1/x` domain** and select with a low-degree comparison —
   expensive, and the last resort.

Note also what Part A/B *agree* on: `1/sqrt` at degree 7 and SiLU at degree 15
are insensitive to the shift strategy, so those two can be built now.

### 7.4 Sequence-length sensitivity — measured, not assumed

The same run at `--tokens=512` (140 s, `oracle/reports/synthetic_8b_t512/`):

| quantity | T = 128 | T = 512 |
|---|---|---|
| `1/x` interval, row-max shift | [1, 30.75] ratio 64.5 | [1, 87.42] ratio **87.4** |
| `1/x` interval, fixed shift | [0.00101, 2.654] ratio 5534 | [0.000518, 5.119] ratio **9874** |
| `1/x` degree, row-max | 63 | **127** |
| `1/x` degree, fixed shift | none ≤ 127 | none ≤ 127 |
| `exp` interval, fixed shift | [−10.03, 0] | [−11.27, 0] |
| `exp` degree | 15 | 15 |
| `1/sqrt` degree | 7 | 7 |
| SiLU degree | 15 | **31** |

Three things to take from this:

* **`1/x` is the fit that does not survive a longer window.** Its degree
  doubled from 128 to 512 tokens even in the *ideal* shift regime, and the
  implementable regime never fits at all. Any reciprocal degree is a function
  of `T` and must be re-measured with `T`.
* **SiLU's degree moved too** (15 → 31), and for a duller reason: four times
  the tokens means four times the samples, so the observed extreme of
  `ffn.gate` is further into the tail. This is caveat 1 of §6 in its most
  concrete form — an interval measured on one window is a *lower bound*, and
  taking it at face value silently under-degrees the fit.
* **`1/sqrt` did not move at all**, at either window. It is the one fit that is
  robust to the sequence length, which is another reason to build RMSNorm
  first.

The `long_window` test vector exists to keep this from being forgotten.

---

## 8. What Sessions 4–7 should do with this

The wave-2 split in `CLAUDE.md` is: slot primitives and RMSNorm; nonlinear
approximations; slot PCMM; CCMM/attention primitives. Mapping those onto the
oracle:

### S4 — slot primitives and RMSNorm

**Inputs.** `L0.x_in` for the norm; for the primitives, any tensor —
`Compare` is shape-generic.

**Expected outputs, in order of increasing difficulty:**

| what to build | oracle reference | tier |
|---|---|---|
| masks, strided sums, broadcasts | construct the host answer directly | `primitive` (1e-5) |
| the channel reduction `Σ_c x²` | `L0.attn_norm.sum_sq` | `primitive` |
| the `1/sqrt` fit alone | `L0.attn_norm.inv_rms`, evaluated over the **rsqrt** interval in §7 | `ThresholdForModule("rmsnorm")` |
| whole RMSNorm | `L0.attn_norm.out` | `ThresholdForModule("rmsnorm")` = 5e-3 |
| residual add | `L0.x_mid` given `L0.x_in` and `L0.o_proj` | `ThresholdForModule("residual")` = **1e-5** |

**Commands.**

```
llama3_oracle_cli dump --tokens=<T> --layers=1 --set=boundaries --out=<dir>
llama3_oracle_cli compare --test=<yours>.tensor \
    --ref=<dir>/L0.attn_norm.out.tensor --module=rmsnorm --axis=0
```

**Watch for.** The `zero_row` and `near_zero_norm` vectors are yours. If the
encrypted norm collapses or explodes there, the *interval's low end* is wrong,
not the circuit. Note also that a residual add has no polynomial in it and is
held at 1e-5 — do not let it inherit a fit-sized tolerance.

### S5 — nonlinear approximations

**Inputs.** The three fit-role tensors and their measured intervals:
`L0.attn_norm.mean_sq` (rsqrt), `L0.attn.exp_input` (exp),
`L0.attn.denominator` (reciprocal), `L0.ffn.gate` (silu).

**Expected outputs.**

| what to build | oracle reference | tier |
|---|---|---|
| `exp` over the measured interval | `L0.attn.exp` (mask already applied) | judged on **err/peak** |
| the denominator reduction | `L0.attn.denominator` | `primitive` (it is a sum) |
| `1/x` over the measured interval | `L0.attn.reciprocal` | judged **pointwise** |
| whole SoftMax | `L0.attn.probs` | `ThresholdForModule("softmax")` = 1e-2, **plus** MaskedResidue and ProbabilityMass |
| SiLU | `L0.ffn.silu` | judged on err/peak, `ThresholdForModule("silu")` = 3e-2 |
| SwiGLU | `L0.ffn.swiglu` | `ThresholdForModule("swiglu")` |

**The one thing to settle first.** §7's shift-strategy comparison. A per-row
maximum is not computable under CKKS; a single global shift is, and it widens
the reciprocal interval by the full spread of the per-row maxima. Decide the
shift strategy — global, per row-block, or an auxiliary refreshed denominator
track — **before** choosing a degree, because the degree follows the interval
and the interval follows the shift.

**Watch for.** Calibrate before fitting. HEonGPU's reciprocal fitted over the
worst case was 98.6% wrong, and calibrating one bound moved a seam from
7.29e-04 to 7.94e-06 — 92×, for free.

### S6 — slot PCMM (projections)

**Inputs.** `L0.attn_norm.out` and `L0.ffn_norm.out` as the activation side;
weights from `MakeSyntheticLayer` or a loaded bundle.

**Expected outputs.** `L0.q_proj`, `L0.k_proj`, `L0.v_proj`, `L0.o_proj`,
`L0.ffn.gate`, `L0.ffn.up`, `L0.ffn.down` — **all at the `primitive` tier
(1e-5)**, because a plaintext matrix product contains no approximation.

**Watch for.**
* Weight matrices here are `[in_features, out_features]`, the transpose of
  HuggingFace's. The transpose is done once at load time and nowhere else.
* K and V are **1024 channels, not 4096** — that asymmetry is GQA and it is the
  most common place to accidentally build MHA.
* `gain` is the criterion that catches a mismanaged rescale in a chained
  projection while `rel_l2` still looks fine.
* A projection chained with itself is the cheapest real test: compare
  `Linear(Linear(x, A), B)` against the oracle at the primitive tier.

### S7 — CCMM and attention primitives

**Inputs.** `L0.q_rope`, `L0.k_rope`, `L0.v_proj`.

**Expected outputs.**

| what to build | oracle reference | tier |
|---|---|---|
| RoPE | `L0.q_rope`, `L0.k_rope` | `ThresholdForModule("rope")` = **1e-5** — no polynomial |
| QKᵀ | `L0.attn.scores` (already scaled by 1/√head_dim) | `primitive` |
| causal mask | `L0.attn.exp` vs `L0.attn.exp_input` | MaskedResidue |
| PV | `L0.attn.context` | `primitive` |
| whole sublayer | `L0.x_mid` | `ThresholdForModule("attention")` = 1.5e-2 |

**Watch for.**
* **GQA, not MHA.** Query head `h` reads key/value head `h / (num_heads /
  num_kv_heads)` — integer division, **not** `h % num_kv_heads`. The oracle
  pins this with `Gqa.HeadsInAGroupShareKeysAndValues` and
  `Gqa.GroupMappingIsNotModulo`; port both, because the modulo mapping produces
  correctly-shaped output and plausible-looking numbers.
* The RoPE convention (§1). `Rope.DependsOnlyOnRelativePosition` is the
  property to port — it is the defining property of rotary embeddings and it
  holds independently of the packing.
* Set `--no-rope` to hit a strictly simpler attention milestone first, but the
  flag disables a part of the **circuit**, not of the model. An attention
  sublayer without RoPE is not Llama-3's, and the record notes RoPE was once
  "wired into nothing" in exactly this way.

---

## 9. What S2 needs from other sessions

| from | what | why |
|---|---|---|
| **S3** | measured per-operation decryption error for Add/Mult/Rescale/HRot/EvalPoly at the levels a module runs at | `ThresholdPrimitive()` is currently a hypothesis anchored only on the bootstrap error. It must be re-anchored on measurement. |
| **S3** | Cheddar's usable dynamic range at the chosen scale | the 1e12 dynamic-range budget in `RecommendDegree` is a placeholder. |
| **S3** | Cheddar `EvalPoly`'s real level cost per degree | the `depth = ceil(log2(deg+1))` column is a structural count, not a measurement of Cheddar. |
| **S0** | the packing, and whether a per-row-block SoftMax shift is expressible in it | it decides the reciprocal interval, which decides the degree. |
| **S0** | the sequence length of the first correctness milestone | every measured interval is a function of `T`. |
| **S0** | decision on `add_subdirectory(oracle)` | §11. |
| **S1** | nothing. The oracle is deliberately independent of the parameter choice. | |
| **whoever has weights** | a nine-file float32 bundle | every range in §7 is synthetic until then. |

---

## 10. Unresolved questions

1. **The SoftMax shift strategy is unresolved and it is the largest open item.**
   §7 measures what a single global shift costs. Nothing here decides what to
   do about it, because the answer depends on the packing.
2. **The multi-block error composition exponent is unmeasured.** `√n` is
   assumed; a correlated error would grow like `n`.
3. **Real weights are not connected.** Every calibration number is synthetic.
   The scaling choice makes the ranges plausible, not correct.
4. **One layer's weights are reused at every layer** when a real bundle is
   loaded, because the bundle format holds one layer. A multi-layer calibration
   on a loaded bundle is therefore not a real 32-layer measurement, and the
   loader says so rather than pretending.
5. **No KV cache.** `GqaAttention` requires `k.tokens == q.tokens` and rejects
   anything else with a message naming the reason. Prefill only. Decode needs a
   separate `k_offset` path.
6. **Llama 3.0 vs 3.1.** If the weights supplied later are 3.1, the missing
   `rope_scaling` makes this oracle wrong. Check before trusting a comparison.
7. **The `rmsnorm_fold_mean_into_fit` flag currently only relabels which
   interval is reported as the rsqrt fit input**; it does not change the
   computed value (the two forms are algebraically identical). If an encrypted
   variant folds `1/hidden` into the fit, calibrate against `sum_sq` rather
   than `mean_sq`.
8. **Cross-host bit-identity is not claimed** and cannot be, because libm
   differs. Use the tolerance-based comparison for anything crossing hosts.

---

## 11. The patch that wires the oracle into the shared build — for S0

Not applied. `oracle/` builds standalone precisely so that S2 does not edit a
shared integration file during wave 1. If the encrypted tests need to link
against the reference, this is the whole change to the root `CMakeLists.txt`:

```cmake
option(BUILD_ORACLE "Build the plaintext Llama-3 reference oracle" ON)
if(BUILD_ORACLE)
  add_subdirectory(oracle)
endif()
```

and then, in `unittest/CMakeLists.txt`, add `llama3_oracle` to the
`target_link_libraries` of whichever test needs it. The oracle target is pure
C++17 with no CUDA, no RMM and no FetchContent, so it adds nothing to configure
time and needs no network.

Note the tension worth flagging: Cheddar's `unittest/` tree fetches googletest
and nlohmann/json from GitHub at configure time. The oracle deliberately does
neither — it carries a ~70-line test framework and a ~100-line JSON writer — so
that it builds on a machine with no network and no GPU. If S0 folds it in,
please keep that property.

---

## 12. What S2 did not do

* **No encrypted kernels.** Nothing in `oracle/` touches Cheddar's `Context`,
  and nothing was added to `src/` or `include/`.
* **No cryptographic parameters.** `parameters/` was not read for any decision
  and not modified.
* **No layout or packing decision.** The oracle's tensor shapes are logical.
* **No GPU.** The oracle is CPU-only by construction; S3 kept the A6000s.
* **No model weights.** Nothing was downloaded, and no weight is committed.
* **No shared build file was edited.** The root `CMakeLists.txt` and
  `unittest/CMakeLists.txt` are untouched; §11 is a patch, not a change.
* **No `--weights-dir` run has ever been performed**, because no bundle exists
  on this machine. The loader is unit-tested against a bundle the test itself
  writes, which proves the format handling but not the real checkpoint.
* **Sicily was not touched** — no directory, no tmux session, no build.
