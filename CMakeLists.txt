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

set(sources
    nn-framework/main.cu
)
add_executable(${CMAKE_PROJECT_NAME} ${sources})

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()