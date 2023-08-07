cmake_minimum_required(VERSION 3.12)
project(interpolate)

# set(CMAKE_PREFIX_PATH "/diskssd/yzhang/miniconda3/envs/mlprod/lib/python3.10/site-packages/pybind11")
set(CMAKE_PREFIX_PATH "/home/yzhang/miniconda3/envs/mlenv/lib/python3.9/site-packages/pybind11")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
# pybind11_add_module(cutils cutils.cpp)
pybind11_add_module(testmodule hello.cpp)
pybind11_add_module(interpolate interpolate.cpp baseutils.cpp)

# add_library(surface surface.cpp baseutils.cpp)

