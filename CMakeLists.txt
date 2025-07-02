cmake_minimum_required(VERSION 3.27)

project(bayesnet
  VERSION 1.2.0
  DESCRIPTION "Bayesian Network and basic classifiers Library."
  HOMEPAGE_URL "https://github.com/rmontanana/bayesnet"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
cmake_policy(SET CMP0135 NEW)

# Global CMake variables
# ----------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED            ON)
set(CMAKE_CXX_EXTENSIONS                  OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS          ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-default-inline")
endif()

# Options
# -------
option(ENABLE_TESTING "Unit testing build"                       OFF)
option(CODE_COVERAGE "Collect coverage from test library"        OFF)

    
find_package(Torch CONFIG REQUIRED)
if(NOT TARGET torch::torch)
    add_library(torch::torch INTERFACE IMPORTED GLOBAL)
    # expose include paths and libraries that the find-module discovered
    set_target_properties(torch::torch PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES      "${TORCH_LIBRARIES}")
endif()

find_package(fimdlp CONFIG REQUIRED)
find_package(folding CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

add_subdirectory(config)

# Add the library
# ---------------
include_directories(
    ${bayesnet_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/configured_files/include
)

file(GLOB_RECURSE Sources "bayesnet/*.cc")

add_library(bayesnet ${Sources})

target_link_libraries(bayesnet
    nlohmann_json::nlohmann_json
    folding::folding
    fimdlp::fimdlp
    torch::torch
    arff-files::arff-files
)

# Testing
# -------
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE("Debug mode")
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE("Release mode")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")
if (ENABLE_TESTING)
    MESSAGE(STATUS "Testing enabled")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage -fno-elide-constructors")
    find_package(Catch2 CONFIG REQUIRED)
    find_package(arff-files CONFIG REQUIRED)
    enable_testing()
    include(CTest)
    add_subdirectory(tests)
else(ENABLE_TESTING)
    
endif (ENABLE_TESTING)

# Installation
# ------------
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/bayesnetConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/bayesnetConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/bayesnetConfig.cmake"
    INSTALL_DESTINATION share/bayesnet)

install(TARGETS bayesnet
        EXPORT bayesnetTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(DIRECTORY bayesnet/ 
    DESTINATION include/bayesnet 
    FILES_MATCHING 
    PATTERN "*.h")
install(FILES ${CMAKE_BINARY_DIR}/configured_files/include/bayesnet/config.h 
    DESTINATION include/bayesnet)

install(EXPORT bayesnetTargets
        FILE bayesnetTargets.cmake
        NAMESPACE bayesnet::
        DESTINATION share/bayesnet)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/bayesnetConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/bayesnetConfigVersion.cmake"
    DESTINATION share/bayesnet
)
# Documentation
# -------------
find_package(Doxygen)
if (Doxygen_FOUND)
  set(DOC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs)
  set(doxyfile_in ${DOC_DIR}/Doxyfile.in)
  set(doxyfile ${DOC_DIR}/Doxyfile)
  configure_file(${doxyfile_in} ${doxyfile} @ONLY)
  doxygen_add_docs(doxygen
      WORKING_DIRECTORY ${DOC_DIR}
    CONFIG_FILE ${doxyfile})
else (Doxygen_FOUND) 
  MESSAGE("* Doxygen not found")
endif (Doxygen_FOUND)
