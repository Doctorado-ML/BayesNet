cmake_minimum_required(VERSION 3.27)

project(bayesnet
  VERSION 1.1.2
  DESCRIPTION "Bayesian Network and basic classifiers Library."
  HOMEPAGE_URL "https://github.com/rmontanana/bayesnet"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
cmake_policy(SET CMP0135 NEW)

# Package manager detection
if(EXISTS ${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
    include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
    set(USING_CONAN TRUE)
    message(STATUS "Using Conan package manager")
else()
    set(USING_CONAN FALSE)
    message(STATUS "Using vcpkg package manager")
endif()

# Find packages - works with both Conan and vcpkg
find_package(Torch CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# These packages might not be available in Conan yet, so we handle them conditionally
if(NOT USING_CONAN)
    find_package(fimdlp CONFIG REQUIRED)
    find_package(folding CONFIG REQUIRED)
else()
    # For Conan, we'll need to either find alternatives or create custom packages
    # For now, we'll look for them and warn if not found
    find_package(fimdlp CONFIG QUIET)
    find_package(folding CONFIG QUIET)
    if(NOT fimdlp_FOUND)
        message(WARNING "fimdlp not found - you may need to create a custom Conan recipe")
    endif()
    if(NOT folding_FOUND)
        message(WARNING "folding not found - you may need to create a custom Conan recipe")
    endif()
endif()

# Global CMake variables
# ----------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED            ON)
set(CMAKE_CXX_EXTENSIONS                  OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS          ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage -fno-elide-constructors")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-default-inline")
endif()

# Options
# -------
option(ENABLE_CLANG_TIDY "Enable to add clang tidy"              OFF)
option(ENABLE_TESTING "Unit testing build"                       OFF)
option(CODE_COVERAGE "Collect coverage from test library"        OFF)
option(INSTALL_GTEST "Enable installation of googletest"         OFF)

add_subdirectory(config)

if (ENABLE_CLANG_TIDY)
    include(StaticAnalyzers) # clang-tidy
endif (ENABLE_CLANG_TIDY)

# Add the library
# ---------------
include_directories(
    ${bayesnet_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/configured_files/include
)

file(GLOB_RECURSE Sources "bayesnet/*.cc")

add_library(bayesnet ${Sources})

# Link libraries conditionally based on package manager
set(BAYESNET_LINK_LIBRARIES "${TORCH_LIBRARIES}")
if(fimdlp_FOUND)
    list(APPEND BAYESNET_LINK_LIBRARIES fimdlp::fimdlp)
endif()
if(folding_FOUND)
    list(APPEND BAYESNET_LINK_LIBRARIES folding::folding)
endif()

target_link_libraries(bayesnet ${BAYESNET_LINK_LIBRARIES})

# Testing
# -------
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE("Debug mode")
    set(ENABLE_TESTING ON)
    set(CODE_COVERAGE ON)
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")
if (ENABLE_TESTING)
    MESSAGE(STATUS "Testing enabled")
    find_package(Catch2 CONFIG REQUIRED)
    
    # Handle arff-files conditionally for different package managers
    if(NOT USING_CONAN)
        find_package(arff-files CONFIG REQUIRED)
    else()
        find_package(arff-files CONFIG QUIET)
        if(NOT arff-files_FOUND)
            message(WARNING "arff-files not found - you may need to create a custom Conan recipe")
        endif()
    endif()
    
    enable_testing()
    include(CTest)
    add_subdirectory(tests)
else(ENABLE_TESTING)
    message("Release mode")
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
        LIBRARY DESTINATION lib
        CONFIGURATIONS Release)

install(DIRECTORY bayesnet/ 
    DESTINATION include/bayesnet 
    FILES_MATCHING 
    CONFIGURATIONS Release 
    PATTERN "*.h")
install(FILES ${CMAKE_BINARY_DIR}/configured_files/include/bayesnet/config.h 
    DESTINATION include/bayesnet 
    CONFIGURATIONS Release)

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
