# XBA2DE performance profiler. Built only when -DENABLE_BENCHMARK=ON.
#
# Deliberately self-contained: it generates its own synthetic data instead of
# reusing the test suite's ARFF loader, so that measuring stays possible while
# tests/TestUtils.cc is still written against the arff-files 1.x API.

include_directories(
    ${bayesnet_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/configured_files/include
)

add_executable(xba2de_profile xba2de_profile.cc)

target_link_libraries(xba2de_profile PRIVATE
    bayesnet
    torch::torch
    nlohmann_json::nlohmann_json
)
