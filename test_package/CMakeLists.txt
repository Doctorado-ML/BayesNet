cmake_minimum_required(VERSION 3.27)
project(test_bayesnet)

set(CMAKE_CXX_STANDARD 17)

find_package(bayesnet REQUIRED)

add_executable(test_bayesnet test_bayesnet.cpp)
target_link_libraries(test_bayesnet bayesnet::bayesnet)