# `oracle/` — the deterministic plaintext reference for Llama-3-8B on Cheddar

This is Session **S2**'s deliverable: the thing an encrypted Cheddar module is
tested *against*. It computes Llama-3-8B in float64 on the CPU, exports every
intermediate tensor at every FHE module boundary, measures the input range of
every function that will have to be approximated by a polynomial, and defines
the error metrics and acceptance thresholds the encrypted tests use.

It contains **no encrypted kernels, no cryptographic parameters, and no model
weights.** It does not touch a GPU and it never reaches the network.

---

## Build and run

The oracle builds **standalone**. It does not participate in the root
`CMakeLists.txt` — see [`docs/ORACLE_HANDOFF.md`](docs/ORACLE_HANDOFF.md) for
the one-line patch that wires it in, which is S0's to apply.

```bash
cmake -S oracle -B oracle/build -DCMAKE_BUILD_TYPE=Release
cmake --build oracle/build --config Release
ctest --test-dir oracle/build -C Release --output-on-failure
```

Requirements: a C++17 compiler and CMake ≥ 3.16. Nothing else — no CUDA, no
RMM, no googletest, no network access at configure time. It builds and runs on
a laptop.

```
oracle/
  include/oracle/   Config Tensor Rng Json Ops Weights Serialize Compare
                    Calibrate Cheby Vectors
  src/              their implementations
  tools/            oracle_main.cpp -> llama3_oracle_cli
  tests/            Check.h/.cpp (a ~70-line test framework) + OracleTest.cpp
  reports/          checked-in calibration output on synthetic weights
  docs/             ORACLE_HANDOFF.md
```

## The command line

```
llama3_oracle_cli config          the verified configuration and its provenance
llama3_oracle_cli conventions     numerical conventions + serialization format
                                  + a copy-paste Python reader
llama3_oracle_cli boundaries      every exported module boundary and its metadata
llama3_oracle_cli thresholds      the acceptance thresholds and their rationale
llama3_oracle_cli vectors         the test-vector catalogue
llama3_oracle_cli weights-howto   how to connect locally available weights
llama3_oracle_cli dump      --out=DIR [--set=minimal|boundaries|full]
llama3_oracle_cli calibrate --out=DIR [--margin=0.05] [--target=1e-4]
llama3_oracle_cli fit       --function=exp --lo=-8 --hi=0 [--target=1e-4]
llama3_oracle_cli selftest        run every test vector and check the invariants
llama3_oracle_cli compare   --test=A.tensor --ref=B.tensor [--tier=...]
llama3_oracle_cli compare   --test-dir=DIR --ref-dir=DIR  [--module=rmsnorm]
```

`compare` exits non-zero on a threshold failure, so it drops straight into
`ctest` or a CI step. In directory mode a boundary present in the reference and
absent from the test directory is reported as MISSING and counts as a failure —
the reference directory *is* the boundary contract.

Shape options apply to every command that runs the model: `--tokens`,
`--layers`, `--seed`, `--reduced` (a fast small shape), `--heads`,
`--kv-heads`, `--head-dim`, `--intermediate`, `--rope=half_split|interleaved`,
`--no-rope`, `--softmax=row_max|fixed_shift|naive`, `--shift`,
`--weights-dir`, `--dtype=f64|f32`.

Two things worth knowing before the first run:

* the **full 8B shape needs about 1.75 GiB of RAM** (the three FFN matrices are
  470 MiB each in float64) and takes a couple of minutes. `--reduced` runs the
  same circuit in well under a second, and every structural property is the
  same at both widths;
* every number that depends on the model's *values* — every calibration range,
  every degree recommendation — is a property of the weights it was measured
  on. With synthetic weights the reports say so on their first line.

## What it is for, concretely

1. **A value contract.** The configuration is verified from repository
   evidence, not from memory; `llama3_oracle_cli config` prints the file and
   line each number came from and lists separately what is assumed.
2. **A reference for every module.** `RmsNorm`, `Linear`, `ApplyRope`,
   `GqaAttention`, `SoftmaxRow`, `Silu`, `SwiGlu`, `FeedForward`,
   `AttentionSublayer`, `DecoderBlock`, `Forward` — each standalone and pure.
3. **Boundary tensors.** 29 named tensors per layer, with metadata, in a
   documented binary format plus a JSON manifest.
4. **Error metrics and thresholds.** `Compare` reports max absolute, max
   relative, RMSE, normalised RMSE, relative L2, cosine, gain and SNR, per
   tensor and per token/channel. `Accept` checks them against four tiers.
5. **Calibration.** The measured input range of the `1/sqrt`, `exp`, `1/x` and
   SiLU approximations, and a Chebyshev degree search that *measures* the fit
   error over those ranges rather than guessing a degree.
6. **Edge cases.** Thirteen named test vectors, seven of them deliberate
   controls.

## Three things the oracle deliberately does not do

* **It does not choose a packing, a level schedule, or a parameter set.** Those
  are S0's and S1's. The oracle fixes values, not layouts.
* **It does not predict encrypted accuracy.** A Chebyshev approximation error
  measured in float64 is a *lower bound* on what the encrypted evaluation
  achieves; CKKS noise is on top of it and only a run on hardware reports it.
* **It does not know Cheddar's real per-operation error.** The primitive-tier
  threshold is anchored on the two measured Cheddar numbers that exist (the
  bootstrap error) and must be re-anchored when S3's tracer lands.

## Connecting real weights

`llama3_oracle_cli weights-howto` prints the full procedure. In short: this
repository never downloads or contains weights; you produce a nine-file
float32 bundle yourself (the script at
`Projects/HEonGPU/benchmark/fetch_llama3_weights.py` already writes exactly the
format the loader reads), then pass `--weights-dir=<bundle>`. A file whose size
does not match the configuration is reported by name with both counts.

The one thing no assertion can check for you is the **RoPE channel pairing** —
HuggingFace weights need `half_split`, an original Meta checkpoint export needs
`interleaved`, and both produce correctly-shaped, correctly-normed tensors.
`llama3_oracle_cli config` explains how to settle it.
