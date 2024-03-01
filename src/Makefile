# Prepare necessary flags for compilation
CUDA_HOME ?= /usr/local/cuda
CUDA_COMPUTE_CAPABILITY ?= sm_80

PYTHON_INC := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC := $(shell python3 -c "import pybind11; print(pybind11.get_include())")


NVCC=nvcc
# NVCCFLAGS=-use_fast_math -std=c++17 --compiler-options "-fPIC" -O3 -arch=$(CUDA_COMPUTE_CAPABILITY)
NVCCFLAGS=-Xcompiler -fPIC -std=c++17 -use_fast_math -O3 -arch=$(CUDA_COMPUTE_CAPABILITY)

CC=g++
PYFLAGS=-I. -I$(PYTHON_INC) -I$(PYBIND_INC) 
CUDA_RUNTIME=-I$(CUDA_HOME)/include/ -L$(CUDA_HOME)/lib64/ -lcudart


interpolate.o: interpolate.cu
		$(NVCC) $(NVCCFLAGS) -c interpolate.cu -o interpolate.o $(PYFLAGS)

cpuutils.o: cpuutils.cpp cpuutils.h
		$(CC) $(NVCCFLAGS) -o cpuutils.o $(PYFLAGS) cpuutils.cpp

gpuutils.o: gpuutils.cu  gpuutils.cuh
		$(NVCC) -dc $(NVCCFLAGS) -o gpuutils.o gpuutils.cu

# Check if PyFlags are needed or not
# run the gpuutils and then compile the marching_observers
marching_observers: marching_observers.cu  marching_observers.h gpuutils.o 
		$(NVCC) -dc $(NVCCFLAGS) -o marching_observers.o  marching_observers.cu
		$(NVCC) -dlink -arch=$(CUDA_COMPUTE_CAPABILITY) -o marching_observers_link.o gpuutils.o marching_observers.o -Xcompiler -fPIC
	
compile: marching_observers gpuutils.o
		echo "Compiled"
		$(CC) $(PYFLAGS) -shared -fPIC -o marching_observers.so marching_observers_py.cpp marching_observers_link.o gpuutils.o marching_observers.o $(CUDA_RUNTIME)


clean:
		rm -f *.o *.so
# voxelize.so: interpolate.o interpolate_py.cpp
# 		$(CC) -fPIC -shared -o voxelize.so interpolate_py.cpp interpolate.o $(PYFLAGS) $(CUDA_RUNTIME)



