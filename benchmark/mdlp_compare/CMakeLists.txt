cmake_minimum_required(VERSION 3.27)

project(mdlp_bench VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
cmake_policy(SET CMP0135 NEW)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(Torch CONFIG REQUIRED)
if(NOT TARGET torch::torch)
    add_library(torch::torch INTERFACE IMPORTED GLOBAL)
    set_target_properties(torch::torch PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES      "${TORCH_LIBRARIES}")
endif()

find_package(fimdlp CONFIG REQUIRED)
find_package(folding CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(arff-files CONFIG REQUIRED)

# BayesNet is compiled from source here, so that the whole library (and not
# only this executable) is built against the fimdlp version under test.
set(BAYESNET_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# bayesnet/config.h is normally produced by the main build; reproduce it here
# with BayesNet's own name and version rather than this project's.
file(READ ${BAYESNET_ROOT}/CMakeLists.txt _bayesnet_cmakelists)
string(REGEX MATCH "project\\(bayesnet[ \t\r\n]+VERSION[ \t\r\n]+([0-9]+\\.[0-9]+\\.[0-9]+)"
       _match "${_bayesnet_cmakelists}")
set(bayesnet_SOURCE_DIR ${BAYESNET_ROOT})
set(PROJECT_NAME bayesnet)
set(PROJECT_VERSION ${CMAKE_MATCH_1})
set(PROJECT_DESCRIPTION "Bayesian Network and basic classifiers Library.")
message(STATUS "BayesNet sources version: ${PROJECT_VERSION}")
configure_file(
    "${BAYESNET_ROOT}/config/config.h.in"
    "${CMAKE_BINARY_DIR}/configured_files/include/bayesnet/config.h" ESCAPE_QUOTES)

include_directories(
    ${BAYESNET_ROOT}
    ${CMAKE_BINARY_DIR}/configured_files/include
)

file(GLOB_RECURSE BayesNet_SOURCES "${BAYESNET_ROOT}/bayesnet/*.cc")

add_executable(mdlp_bench mdlp_bench.cc ${BayesNet_SOURCES})
target_link_libraries(mdlp_bench PRIVATE
    torch::torch
    fimdlp::fimdlp
    folding::folding
    nlohmann_json::nlohmann_json
    arff-files::arff-files
)
