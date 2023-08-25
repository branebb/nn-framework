cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(nn-framework LANGUAGES CXX CUDA)
message("Project name: " ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(MSVC)
    set(_CRT_SECURE_NO_WARNINGS 1)
    set(CURL_STATICLIB 1)
endif()

include_directories(
    /
)

add_executable(${CMAKE_PROJECT_NAME} ${sources})

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    nn-framework/main.cu
    nn-framework/utils/error_check_cuda.cu
    nn-framework/cudas/layers/relu_activation.cu
    nn-framework/cudas/layers/sigmoid_activation.cu
    nn-framework/cudas/layers/tanh_activation.cu
    nn-framework/cudas/structures/dimensions.cu
    nn-framework/cudas/structures/matrix.cu
    nn-framework/cudas/structures/neural_network.cu
    nn-framework/cudas/cost_functions/MSEcost.cu
    nn-framework/cudas/layers/linear_layer.cu
    nn-framework/coordinates_test.cu
    nn-framework/cudas/layers/softmax_activation.cu
    nn-framework/cudas/optimizers/gradient.cu
    nn-framework/cudas/optimizers/adam.cu
    
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()