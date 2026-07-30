cmake_minimum_required(VERSION 3.16)

project(Project_Template)

set(CMAKE_CXX_STANDARD 17)

add_executable(
    Project_Template
    src/main.cpp
    src/Player.cpp
)

target_include_directories(
    Project_Template
    PRIVATE include
)