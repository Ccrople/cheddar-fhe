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

if (ENABLE_EXTENSION)
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
endif()

configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_30.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_35.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_40.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_40_64bit.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_40_64bit.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ringdegree12_28.json
               ${CMAKE_CURRENT_BINARY_DIR}/ringdegree12_28.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ringdegree12_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/ringdegree12_30.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/ringswitch16_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/ringswitch16_30.json COPYONLY)
