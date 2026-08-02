
cmake_minimum_required(VERSION 3.15)
project(fast_options LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(fast_options fast_options.cpp)
target_compile_features(fast_options PRIVATE cxx_std_17)

if (MSVC)
    target_compile_options(fast_options PRIVATE /O2)
else()
    target_compile_options(fast_options PRIVATE -O3 -march=native)
endif()
