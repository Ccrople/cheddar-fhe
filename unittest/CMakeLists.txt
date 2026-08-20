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

# RingSwitch -> ModDecomp -> PC-MM -> ModPack -> Rescale -> SwitchBack, in one
# process. The only place the chain's level and scale budget is checked.
add_executable(pipeline_chain_test PipelineChainTest.cpp)
target_link_libraries(pipeline_chain_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

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

  # Slots <-> SinC, homomorphically: the last stages of SlotToCoeff and nothing
  # else. One level against SlotToCoeff's three, and the piece the batch CC-MM
  # needs before it can take an operand that arrived encrypted.
  add_executable(sinc_transform_test SinCTransformTest.cpp)
  target_link_libraries(sinc_transform_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

  add_executable(boot_test Bootstrapping.cpp)
  target_link_libraries(boot_test
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

  # Level-targeted bootstrapping: one landing level per fixture.
  add_executable(boot_level_test BootLevelTest.cpp)
  target_link_libraries(boot_level_test
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

  # One whole decoder block through six turns of that cycle. The plan and the
  # calibration tests are host arithmetic; the third bootstraps ninety-six
  # times. All three need LLAMA3_REAL_DIR except the first, and the matrix
  # products are a host stand-in -- see the file header.
  add_executable(llama_block_test LlamaBlockTest.cpp)
  target_link_libraries(llama_block_test
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
