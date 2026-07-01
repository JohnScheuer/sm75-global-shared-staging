cmake_minimum_required(VERSION 3.18)
project(cp_async_microbench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 2070 = SM75
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(bench src/main.cu)

set_target_properties(bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)