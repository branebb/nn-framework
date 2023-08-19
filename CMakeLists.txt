cmake_minimum_required(VERSION 3.17)

project(nn-framework LANGUAGES CXX CUDA)
message("Project name: " ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# set(SOURCES
#     nn-framework/main.cpp
# )

if(MSVC)
    set(_CRT_SECURE_NO_WARNINGS 1)
    set(CURL_STATICLIB 1)
endif()

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ALL)

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    nn-framework/main.cu
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: >
)