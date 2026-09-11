include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  jsoncpp
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(jsoncpp)

add_compile_definitions(
  PARAM_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

add_executable(basic_test BasicTest.cpp)
target_link_libraries(basic_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# The conjugate-invariant ring (Kim and Song, ISISC 2018), which is where
# [SYLPH]'s N real slots come from. Checks the ring's own multiplication --
# Toeplitz plus Hankel, not negacyclic -- against a host reference, and against
# the ordinary ring's product so that a half-applied change cannot pass.
# Degree 4096, so its own binary.
add_executable(ci_ring_test CiRingTest.cpp)
target_link_libraries(ci_ring_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# The conjugate-invariant slot encoding, which is what the ring is for: N real
# slots against the ordinary ring's N/2 complex ones. Validated against the
# canonical embedding summed directly on the host, so the encoder's own
# transform is not its own reference.
add_executable(ci_slot_test CiSlotTest.cpp)
target_link_libraries(ci_slot_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# Coefficient-domain encoding, for the reduction-based matrix products.
add_executable(coeff_encode_test CoeffEncodeTest.cpp)
target_link_libraries(coeff_encode_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# Slots-in-Coefficients, the encoding [KANG]'s batch CC-MM operates in. Placed
# next to the encoding tests rather than the matrix-product ones on purpose: it
# is an encoding, it spends no levels, and the CC-MM work builds upward from
# here. Degree 4096, so its own executable.
add_executable(sinc_encode_test SinCEncodeTest.cpp)
target_link_libraries(sinc_encode_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# [KANG] Algorithm 1, batch plaintext-ciphertext matrix multiplication over the
# subring. Stands on the encoding above and shares its engine with the batch
# CC-MM. Degree 4096, so its own executable.
add_executable(batch_cpmm_test BatchCpmmTest.cpp)
target_link_libraries(batch_cpmm_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# [KANG] Algorithm 2, the TWEAK transform, plus the monomial arithmetic it runs
# on -- the key-free and level-free half of CMT. Degree 4096, its own binary.
add_executable(tweak_test TweakTest.cpp)
target_link_libraries(tweak_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# [KANG] Algorithm 3, the whole CMT: ScrambleAuto on top of TWEAK. Needs
# rotation keys, unlike everything above it. Degree 4096, its own binary.
add_executable(cmt_test CmtTest.cpp)
target_link_libraries(cmt_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# [KANG] Algorithm 4 step 2: the subring matrix product, transform path
# checked against the direct convolution. Degree 4096, its own binary.
add_executable(subring_ct_matrix_test SubringCtMatrixTest.cpp)
target_link_libraries(subring_ct_matrix_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# [KANG] Algorithm 4, the whole batch CC-MM. Degree 4096, its own binary.
add_executable(batch_ccmm_test BatchCcmmTest.cpp)
target_link_libraries(batch_ccmm_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# Bae et al. plaintext-ciphertext matrix product (RLWE format).
add_executable(pcmm_test PcmmTest.cpp)
target_link_libraries(pcmm_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# RLWE -> MLWE decomposition (Bae et al. appendix A).
add_executable(mlwe_test MlweTest.cpp)
target_link_libraries(mlwe_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# The same decomposition on the conjugate-invariant ring, where it is a banded
# map rather than a stride gather. Validated against the ring's own product on
# a tiny host ring, entry by entry against the host scan at the real degree,
# and end to end through ModPack's embedded-secret key switches.
add_executable(ci_mlwe_test CiMlweTest.cpp)
target_link_libraries(ci_mlwe_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# Does the NTT work below log_degree 16? Sylph needs degree 4096 for the batch
# CC-MM. MUST be run one --gtest_filter per process: __cm_log_degree is
# write-once per process, so a second degree in the same process would silently
# run at the first one's. See run_small_ring_ntt.sh.
add_executable(small_ring_ntt_test SmallRingNttTest.cpp)
target_link_libraries(small_ring_ntt_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# A full Context at ring degree 4096 (Sylph's batch CC-MM ring, KANG S12).
# Its own executable for the same reason as above: appending the degree-4096
# preset to an existing suite would silently evaluate it at degree 65536.
add_executable(small_ring_context_test SmallRingContextTest.cpp)
target_link_libraries(small_ring_context_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# The PC-MM against a real Llama-3-8B layer-2 query projection. Needs the
# weight bundle; set LLAMA3_REAL_DIR or the test skips itself. Degree 4096, so
# again its own executable.
add_executable(llama_pcmm_test LlamaPcmmTest.cpp)
target_link_libraries(llama_pcmm_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# Two ring degrees alive in one process. This is the test the multi-ring change
# exists to pass, and the one case that must NOT be split per process.
add_executable(multi_ring_test MultiRingTest.cpp)
target_link_libraries(multi_ring_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# Ring switching, 65536 -> 16 x 4096. Needs three rings' worth of parameter
# files and, like the above, more than one Context per process.
add_executable(ring_switch_test RingSwitchTest.cpp)
target_link_libraries(ring_switch_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# The keyless lift of a conjugate-invariant ciphertext into the ordinary ring
# of the same conductor, and the keyless descent back (Doing.md 1.5bk).
add_executable(ci_lift_test CiLiftTest.cpp)
target_link_libraries(ci_lift_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# The conjugate-invariant FFN, one stage at a time: the crossing, RMSNorm on
# the half-density image a projection emits, SiLU and the two products. Its
# own binary because ci_switched_ccmm_test's converter builds dominate that
# suite and none of them are needed here.
add_executable(ci_ffn_test CiFfnTest.cpp)
target_link_libraries(ci_ffn_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# ci16 -> RingSwitch -> CiLift -> [KANG] Alg. 4 -> CiDescend -> SwitchBack:
# the conjugate-invariant CC-MM chain end to end, three rings alive at once.
add_executable(ci_switched_ccmm_test CiSwitchedCcmmTest.cpp)
target_link_libraries(ci_switched_ccmm_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# RingSwitch -> ModDecomp -> PC-MM -> ModPack -> Rescale -> SwitchBack, in one
# process. The only place the chain's level and scale budget is checked.
add_executable(pipeline_chain_test PipelineChainTest.cpp)
target_link_libraries(pipeline_chain_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# [SYLPH] table 4's CC-MM row: 8 x 65536 -> RingSwitch -> 128 x 4096 ->
# [KANG] Algorithm 4 -> SwitchBack, at Llama-3's per-head 128 x 128 shape.
# Two rings alive at once, so it is its own binary like the two above.
add_executable(switched_ccmm_test SwitchedCcmmTest.cpp)
target_link_libraries(switched_ccmm_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# A slot field traded for the ciphertext index: the cross-ciphertext move the
# score product's right operand needs, because its column axis is the key token
# and no weight matrix can place a token. One ring, one level.
add_executable(ct_axis_exchange_test CtAxisExchangeTest.cpp)
target_link_libraries(ct_axis_exchange_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

if (ENABLE_EXTENSION)
  # The block's packing all the way into a score product: SlotPermute, then
  # SlotToSinC, then the switched CC-MM. Three Contexts and one secret between
  # the two big ones, so it is its own binary and needs the extension for
  # BootContext and SlotPermute.
  add_executable(attention_transport_test AttentionTransportTest.cpp)
  target_link_libraries(attention_transport_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

endif()

if (ENABLE_EXTENSION)
  # A Llama-3 projection, encrypted, in the block's own packing on
  # bootparam_35, with no ring switch. ModDecomp isolates the channel axis
  # because the block's slot layout bit-reverses the channel into the low
  # coefficient bits; see the file header. Needs the extension only for
  # AttentionPacking -- it undefines ENABLE_EXTENSION so Testbed builds a plain
  # Context and no bootstrapping key is generated.
  add_executable(llama_projection_test LlamaProjectionTest.cpp)
  target_link_libraries(llama_projection_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # The same projection through [SYLPH] section 3.2's own descent -- ring switch
  # first, then ModDecomp at rank 16 -- run against the direct route on the same
  # ciphertexts. Three rings alive at once, so it is its own binary.
  add_executable(switched_projection_test SwitchedProjectionTest.cpp)
  target_link_libraries(switched_projection_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Slots <-> SinC, homomorphically: the last stages of SlotToCoeff and nothing
  # else. One level against SlotToCoeff's three, and the piece the batch CC-MM
  # needs before it can take an operand that arrived encrypted.
  # An arbitrary slot permutation as one LinearTransform -- the transpose
  # [SYLPH] 3.2 says survives its layout choice, and the move the CC-MM's
  # operands need to get the token off the low slot bits.
  add_executable(slot_permute_test SlotPermuteTest.cpp)
  target_link_libraries(slot_permute_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  add_executable(sinc_transform_test SinCTransformTest.cpp)
  target_link_libraries(sinc_transform_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  add_executable(boot_test Bootstrapping.cpp)
  target_link_libraries(boot_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # One ciphertext's bootstrap latency, synced on both sides of every
  # repetition -- the number `boot_test`'s Bootstrap prints is a slice of a
  # pipelined queue, which is not what a single bootstrap costs.
  add_executable(boot_bench BootBench.cpp)
  target_link_libraries(boot_bench
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # CoeffToSlot and SlotToCoeff on the real subring, against the encoder rather
  # than against another run of the transform. Compiles the transforms
  # standalone, since BootContext still refuses the conjugate-invariant flag,
  # and runs unchanged on bootparam_35 as the control.
  add_executable(ci_fft_test CiFftTest.cpp)
  target_link_libraries(ci_fft_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # The same two transforms in the MODULE basis (Doing.md 3.5): StC lands the
  # slots as module coordinates -- checked by the host scan and by the device
  # ModDecomp/ModPack round trip -- and CtS reads them back through the pair
  # correction, against the native CtS on the same element as the control.
  add_executable(ci_module_basis_test CiModuleBasisTest.cpp)
  target_link_libraries(ci_module_basis_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # The SinC conversions fused into the bootstrap on R+ (Doing.md 3.16): the
  # tower basis's forward, CtS' and prefix on the device against the host
  # encoder, and the fused return (HalfBootTower + prefix) on the K = 64
  # landing ring under the tower-sparse secret.
  add_executable(ci_sinc_basis_test CiSinCBasisTest.cpp)
  target_link_libraries(ci_sinc_basis_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # The BATCHED layer on R+ (branch llama3_CI_new): one channel per
  # ciphertext, the slots holding (token, instance) pairs, and every
  # projection as [KANG] Algorithm 1 with a batch-invariant weight -- the
  # int8 GEMM over whole ciphertexts, no ModDecomp/ModPack, no key. Against a
  # host reference at a small width, then the real layer-0 weights at the
  # model's width (LLAMA3_ALL_DIR).
  add_executable(ci_batch_test CiBatchTest.cpp)
  target_link_libraries(ci_batch_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # BERT-Tiny on the batched layout (branch BERT_tiny): one ring, the
  # diagonal attention products, Cho softmax, LayerNorm and GELU per
  # ciphertext; layer and chain against bert_tiny/reference.py's h_L{k}.f64
  # with bert_tiny/sim.py's calib.json (BERT_TINY_ALL / BERT_TINY_REF).
  add_executable(ci_bert_tiny_test CiBertTinyTest.cpp)
  target_link_libraries(ci_bert_tiny_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # S3 capability probes. Additive only; nothing above is modified.
  # These need ENABLE_EXTENSION for EvalPoly / LinearTransform / BootContext.
  add_executable(capability_probe CapabilityProbe.cpp)
  target_link_libraries(capability_probe
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  add_executable(chain_probe ChainProbe.cpp)
  target_link_libraries(chain_probe
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Llama RMSNorm on the real layer-2 residual stream. Needs ENABLE_EXTENSION
  # for EvalPoly, and LLAMA3_REAL_DIR for the weight bundle.
  add_executable(rms_norm_test RmsNormTest.cpp)
  target_link_libraries(rms_norm_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Llama SiLU. The fit and sweep tests need only EvalPoly; the real-weight
  # test needs LLAMA3_REAL_DIR and skips without it.
  add_executable(silu_test SiLuTest.cpp)
  target_link_libraries(silu_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Llama SoftMax, [SYLPH] section 2.3.
  add_executable(softmax_test SoftMaxTest.cpp)
  target_link_libraries(softmax_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # [SYLPH] section 3.4 / Algorithm 1: the slim polynomial evaluation. The
  # decomposition half is pure host arithmetic and runs without a GPU; the
  # Algorithm 1 half needs one.
  add_executable(slim_poly_test SlimPolyTest.cpp)
  target_link_libraries(slim_poly_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # [SYLPH] section 4.2 eq. (5) and appendix E lemma 2: the depth-one PCMM.
  # The index algebra runs without a GPU.
  add_executable(sylph_pcmm_test SylphPcmmTest.cpp)
  target_link_libraries(sylph_pcmm_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Level-targeted bootstrapping: one landing level per fixture.
  add_executable(boot_level_test BootLevelTest.cpp)
  target_link_libraries(boot_level_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Landing (flexible-output) bootstrap on the CI ring: a sub-ladder of
  # ci16_35 that climbs less and lands lower, and crosses keylessly from
  # ci16_35. Uses RingFixture (two rings, one secret), not Testbed.
  add_executable(boot_landing_test BootLandingTest.cpp)
  target_link_libraries(boot_landing_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # One parameter set, any landing: the runtime ladder (`extension/
  # LandingLadder`, the ci16_42 pools cut for a landing at run time) against
  # its host mirror, the keyless crossing between the pool and the ladder it
  # cuts, and the climb's cost. Two rings, one secret (RingFixture).
  add_executable(landing_ladder_test LandingLadderTest.cpp)
  target_link_libraries(landing_ladder_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # The robust-parameter acceptance test: the scale/landing contract measured
  # against the library's nominal numbers, one preset per process
  # (CHEDDAR_ROBUST_PARAM; CHEDDAR_ROBUST_STRICT=1 is the v3 contract). The
  # host half of the pair is reference/scripts/param_audit.py.
  add_executable(param_robust_test ParamRobustTest.cpp)
  target_link_libraries(param_robust_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Llama RoPE, [SYLPH] section 3.2. The real-weight case needs LLAMA3_REAL_DIR.
  add_executable(rope_test RoPeTest.cpp)
  target_link_libraries(rope_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Online timings for the non-linear operators. Performance, not correctness.
  add_executable(profile_test ProfileTest.cpp)
  target_link_libraries(profile_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # PCMM setup/online split and the plaintext footprint.
  add_executable(pcmm_profile_test PcmmProfileTest.cpp)
  target_link_libraries(pcmm_profile_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # The cuBLAS PCMM against the hand kernel. Builds either way; the test body
  # compiles out without USE_CUBLAS.
  add_executable(pcmm_blas_test PcmmBlasTest.cpp)
  target_link_libraries(pcmm_blas_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # HalfBoot: the only route from coefficient encoding back to slots.
  add_executable(half_boot_test HalfBootTest.cpp)
  target_link_libraries(half_boot_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # What HalfBoot leaves on the message, against the level-zero prime ratio it
  # is derived from. Three presets, because the nominal power of two is out by
  # a different amount on each.
  add_executable(crossing_constant_test CrossingConstantTest.cpp)
  target_link_libraries(crossing_constant_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # [SYLPH] 3.2's attention packing layout, and the GPU measurement of the
  # coefficient-to-slot permutation it is built around. The host tests need no
  # GPU; the measurement runs 17 HalfBoots, so filter to it deliberately.
  add_executable(attention_packing_test AttentionPackingTest.cpp)
  target_link_libraries(attention_packing_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # The gap between EvalMod and StC that [SYLPH] figure 2's schedule needs.
  add_executable(slack_schedule_test SlackScheduleTest.cpp)
  target_link_libraries(slack_schedule_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # [SYLPH] figure 2's cycle assembled: the level and scale bookkeeping between
  # the slot and coefficient legs. ThePlanClosesForAWholeBlock is arithmetic
  # only; the other two bootstrap, and the third needs LLAMA3_REAL_DIR.
  add_executable(sylph_schedule_test SylphScheduleTest.cpp)
  target_link_libraries(sylph_schedule_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # [32] section 5.1's three claims for the conjugate-invariant ring -- same
  # ciphertext, twice the slots, same arithmetic -- checked on ci16_35 against
  # bootparam_35, which differ in nothing but the ring. Two Contexts in one
  # process, so it cannot use Testbed.
  add_executable(ci_parity_test CiParityTest.cpp)
  target_link_libraries(ci_parity_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Llama-3-8B decoder layers at the MODEL'S OWN WIDTH, encrypted end to end,
  # one after another: 4096 live channels in seventeen half-density
  # ciphertexts, the real parameters of a named layer, and the output against
  # the same layer in float64. Needs LLAMA3_ALL_DIR and LLAMA3_REF_DIR;
  # CHEDDAR_CI_LAYERS says how many.
  add_executable(ci_model_test CiModelTest.cpp)
  target_link_libraries(ci_model_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # `CiLlamaSeam` -- the promotion of the seam that the layer and three cases in
  # CiFfnTest each wrote out inline. Checked the way its consumer reads it:
  # coefficients out, then the banded scan ModDecomp inverts, live band and
  # dead band apart, with the one-token-out reading as a control that must
  # fail. About two and a half minutes, against a quarter of an hour for the
  # layer.
  add_executable(ci_seam_test CiSeamTest.cpp)
  target_link_libraries(ci_seam_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Keys, ciphertexts and compiled transforms written to disk and read back.
  # Every case decrypts through to a message and compares it with a value
  # computed on the host -- the rotation case uses the LOADED map in isolation,
  # so a key that survived the bytes but not its index cannot pass. Small
  # rings; the subject is the format, not the arithmetic.
  add_executable(serialization_test SerializationTest.cpp)
  target_link_libraries(serialization_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # Per-object device memory for the CI layer's bootstrap sets, through RMM's
  # statistics adaptor. Alone in its binary on purpose: the adaptor has to be
  # installed before the first Context, so a second test in the process would
  # find the request refused.
  add_executable(memory_ledger_test MemoryLedgerTest.cpp)
  target_link_libraries(memory_ledger_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)
endif()

configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_30.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_35.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_40.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_40_64bit.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_40_64bit.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ringdegree12_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/ringdegree12_30.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ringswitch16_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/ringswitch16_30.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/sylphflow16_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/sylphflow16_35.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/sylphflow16_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/sylphflow16_40.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ringdegree12_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/ringdegree12_35.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ringswitch16_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/ringswitch16_35.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ci12_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/ci12_30.json COPYONLY)
# THE 2^35 FAMILY (2026-09-11, `reference/scripts/ci35_family.py`): three
# pools on ci16_35's own compute prefix, one per K, each with an EvalMod band
# of stationary 2^58 pairs so that `LandingLadder` (`CHEDDAR_BOOT_LANDING=L`
# in the fixtures, `CHEDDAR_CI_{BASE,FFN,LEG}_LANDING` in ci_model_test) cuts
# any landing at run time. They replace ci16_35_k16_w58.json and every
# `ci16_35_land*` / `ci16_35_stc2` file:
#   ci16_35_k16_w58  K 16, dec 19, four CtS levels  -- the base ring (ci16_35's shape)
#   ci16_35_k32_w58  K 32, dec 19, two CtS levels   -- the FFN ring's pool
#                    (cut at 13 = the old land13c2e9, CtS [50, 50])
#   ci16_35_k64_w58  K 64, dec 17, three CtS levels -- the tower ring
#                    (the old land17c3e10v3; cut at 13 = the old land13c3e10v3)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ci16_35_k16_w58.json
               ${CMAKE_CURRENT_BINARY_DIR}/ci16_35_k16_w58.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ci16_35_k32_w58.json
               ${CMAKE_CURRENT_BINARY_DIR}/ci16_35_k32_w58.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ci16_35_k64_w58.json
               ${CMAKE_CURRENT_BINARY_DIR}/ci16_35_k64_w58.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ci16_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/ci16_40.json COPYONLY)
# The 2^42 pools (CI_PARAM_20BIT.md): the first cut, one pool per K with
# 20.9 / 19.6 / 18.3 bits after Boot and `LandingLadder` cutting any landing
# at run time; and the second cut (`reference/scripts/ci42_family.py`), the
# same on ONE compute prefix at the levels the Llama layer needs.
foreach(P ci16_42_k16_w60 ci16_42_k32_w60 ci16_42_k64_w60
          ci16_42_k16 ci16_42_k32 ci16_42_k64)
  configure_file(${CMAKE_SOURCE_DIR}/parameters/${P}.json
                 ${CMAKE_CURRENT_BINARY_DIR}/${P}.json COPYONLY)
endforeach()
configure_file(${CMAKE_SOURCE_DIR}/parameters/ci_ringswitch16_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/ci_ringswitch16_35.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ci12_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/ci12_35.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ringdegree13_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/ringdegree13_35.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ci12_35_boot.json
               ${CMAKE_CURRENT_BINARY_DIR}/ci12_35_boot.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ci_ringswitch16_35_boot.json
               ${CMAKE_CURRENT_BINARY_DIR}/ci_ringswitch16_35_boot.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ringdegree13_35_boot.json
               ${CMAKE_CURRENT_BINARY_DIR}/ringdegree13_35_boot.json COPYONLY)

# The encoding unit on the GPU, against the host encoder it replaces. Checks
# the coefficient and matrix encodings limb by limb -- both routes round the
# same way, so agreement is not the bar, identity is -- and then measures every
# stage with CUDA events, exact byte counts and the occupancy calculator.
add_executable(encode_gpu_test EncodeGpuTest.cpp)
target_link_libraries(encode_gpu_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

# The projection leg fed the model's tensor on the device, against itself fed
# the declared double matrix: the same words out, on both product paths.
add_executable(device_weights_test DeviceWeightsTest.cpp)
target_link_libraries(device_weights_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

if (ENABLE_EXTENSION)
  # Two compilations of one transform, coefficient by coefficient through the
  # inverse NTT: a transform compiled on the device against the host encoder
  # on a small ring, and -- given two converter cache files -- the leg's real
  # converters as the host wrote them against the device's.
  add_executable(converter_compare_test ConverterCompareTest.cpp)
  target_link_libraries(converter_compare_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # The batched layer on the PLAIN slot map, with the chain addressing folded
  # into the CC-MM's forward as a `forward_premap`: the premap is the block
  # index's bit reversal, and the fold must cost nothing (same BSGS split,
  # same diagonals, same plaintext bytes) and compute the same operand. What
  # the plain map buys is a PC-attention plaintext that is sparsely packed
  # (512 coefficients of 65536) instead of dense.
  add_executable(pc_premap_test PcPremapTest.cpp)
  target_link_libraries(pc_premap_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  # The first CONSUMER of the plain map: [SYLPH] section 4's public-context
  # attention, which under that map is [KANG] Algorithm 1 twice with one exp
  # between -- the encrypted queries against a public KV cache that differs
  # per instance, streamed so the residency does not follow the context
  # length.
  add_executable(pc_attention_test PcAttentionTest.cpp)
  target_link_libraries(pc_attention_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  add_executable(public_prefill_test PublicPrefillTest.cpp)
  target_link_libraries(public_prefill_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)
endif()
