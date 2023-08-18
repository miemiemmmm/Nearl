cmake_minimum_required(VERSION 3.12)
project(interpolate)

# set(CMAKE_PREFIX_PATH "/diskssd/yzhang/miniconda3/envs/mlenv/lib/python3.9/site-packages/pybind11")
# set(CMAKE_PREFIX_PATH "/diskssd/yzhang/miniconda3/envs/mlprod/lib/python3.10/site-packages/pybind11")
# set(CMAKE_PREFIX_PATH "/home/yzhang/miniconda3/envs/mlenv/lib/python3.9/site-packages/pybind11")

# set(CMAKE_PREFIX_PATH "${PYBIND11_INCLUDE_DIR}")
# add_subdirectory(pybind11)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER pgc++)


find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add OpenACC compiler flags
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenacc")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenacc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -Minfo=accel")

find_package(pybind11 REQUIRED)
find_package(OpenMP)

# pybind11_add_module(cutils cutils.cpp)
pybind11_add_module(testmodule hello.cpp)
# pybind11_add_module(interpolate interpolate.cpp baseutils.cpp)
pybind11_add_module(interpolate baseutils.cpp interpolate.cpp)

if(OPENMP_FOUND)
    target_link_libraries(testmodule PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(interpolate PUBLIC OpenMP::OpenMP_CXX)
endif()


# add_library(surface surface.cpp baseutils.cpp)

