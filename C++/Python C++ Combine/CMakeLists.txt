cmake_minimum_required(VERSION 3.16)

project(cpp_python_example)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

pybind11_add_module(
    fast_code
    cpp/fast_code.cpp
)

# Put the compiled Python module beside main.py
set_target_properties(
    fast_code
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)