# XBA2DE performance profiler. Built only when -DENABLE_BENCHMARK=ON.
#
# Loads the real datasets itself rather than through tests/TestUtils.h: the
# class column is taken from tests/data/all.txt, because kdd_JapaneseVowels
# declares its class first and RawDatasets assumes it is last.

find_package(arff-files CONFIG REQUIRED)

include_directories(
    ${bayesnet_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/configured_files/include
)

add_executable(xba2de_profile xba2de_profile.cc)

target_link_libraries(xba2de_profile PRIVATE
    bayesnet
    torch::torch
    fimdlp::fimdlp
    arff-files::arff-files
    nlohmann_json::nlohmann_json
)
