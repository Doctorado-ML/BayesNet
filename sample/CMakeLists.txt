cmake_minimum_required(VERSION 3.20)

project(bayesnet_sample VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)

find_package(Torch CONFIG REQUIRED)
find_package(fimdlp CONFIG REQUIRED)
find_package(folding CONFIG REQUIRED)
find_package(arff-files CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

option(BAYESNET_VCPKG_CONFIG "Use vcpkg config for BayesNet" ON)

if (BAYESNET_VCPKG_CONFIG)
  message(STATUS "Using BayesNet vcpkg config")
  find_package(bayesnet CONFIG REQUIRED)
  set(BayesNet_LIBRARIES bayesnet::bayesnet)
else(BAYESNET_VCPKG_CONFIG)
  message(STATUS "Using BayesNet local library config")
  find_library(bayesnet NAMES libbayesnet bayesnet libbayesnet.a PATHS ${Platform_SOURCE_DIR}/../lib/lib REQUIRED)
  find_path(Bayesnet_INCLUDE_DIRS REQUIRED NAMES bayesnet PATHS ${Platform_SOURCE_DIR}/../lib/include)
  add_library(bayesnet::bayesnet UNKNOWN IMPORTED)
  set_target_properties(bayesnet::bayesnet PROPERTIES
    IMPORTED_LOCATION ${bayesnet}
    INTERFACE_INCLUDE_DIRECTORIES ${Bayesnet_INCLUDE_DIRS}
  )
endif(BAYESNET_VCPKG_CONFIG)
message(STATUS "BayesNet: ${bayesnet}")

add_executable(bayesnet_sample sample.cc) 
target_link_libraries(bayesnet_sample PRIVATE 
    fimdlp::fimdlp 
    arff-files::arff-files 
    "${TORCH_LIBRARIES}" 
    bayesnet::bayesnet
    folding::folding
)
