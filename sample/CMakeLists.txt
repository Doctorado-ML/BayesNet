cmake_minimum_required(VERSION 3.20)

project(bayesnet_sample VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch CONFIG REQUIRED)
find_package(fimdlp CONFIG REQUIRED)
find_package(folding CONFIG REQUIRED)
find_package(arff-files CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(bayesnet CONFIG REQUIRED)

add_executable(bayesnet_sample sample.cc) 
target_link_libraries(bayesnet_sample PRIVATE 
    fimdlp::fimdlp 
    arff-files::arff-files
    torch::torch
    bayesnet::bayesnet
    folding::folding
    nlohmann_json::nlohmann_json
)
