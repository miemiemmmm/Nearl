cmake_minimum_required(VERSION 3.12)
project(nearl_static LANGUAGES C CXX)

# With the `option` clause, this variable could de defined by user. OFF by Default. 
option(COMPILE_WITH_GPU "Build with GPU support" ON)



if (COMPILE_WITH_GPU)
  set(CMAKE_CXX_COMPILER "${OPENACC_COMPILER}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -Minfo=accel")
else ()
	find_package(OpenMP)
	set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wall -Wextra")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Let the compiler know where to find the PyBind11
# set(CMAKE_PREFIX_PATH "${PYBIND11_INCLUDE_DIR}")

add_subdirectory("${PYBIND11_DIR}" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/_deps/pybind11-build)

# pybind11_add_module(cutils cutils.cpp)
# pybind11_add_module(testmodule hello.cpp)
pybind11_add_module(interpolate interpolate.cpp baseutils.cpp)
# pybind11_add_module(interpolate baseutils.cpp interpolate.cpp)

if(OPENMP_FOUND)
	# target_link_libraries(testmodule PUBLIC OpenMP::OpenMP_CXX)
	target_link_libraries(interpolate PUBLIC OpenMP::OpenMP_CXX)
endif()


# add_library(surface surface.cpp baseutils.cpp)

