# Prepare default environment variable for compilation
CUDA_HOME ?= /usr/local/cuda
CUDA_COMPUTE_CAPABILITY ?= sm_80

# Python-related variables
# PYTHON_INC := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
# PYBIND_INC := $(shell python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INC := $(shell python3-config --includes)
PYTHON_TAIL := $(shell python3-config --extension-suffix)
PYBIND_INC := $(shell python3 -m pybind11 --includes)

# Compiler and compiler flags
NVCC=nvcc
# NVCCFLAGS=-use_fast_math -std=c++17 --compiler-options "-fPIC" -O3 -arch=$(CUDA_COMPUTE_CAPABILITY)
NVCCFLAGS=-Xcompiler -fPIC -std=c++17 -use_fast_math -O3 -arch=$(CUDA_COMPUTE_CAPABILITY)

CC=g++
PYFLAGS=-I. $(PYTHON_INC) $(PYBIND_INC) 
CUDA_RUNTIME=-I$(CUDA_HOME)/include/ -L$(CUDA_HOME)/lib64/ -lcudart


gpuutils: gpuutils.cu gpuutils.cuh
		$(NVCC) $(NVCCFLAGS) -c gpuutils.cu -o gpuutils.o 

marching_observers: marching_observers.cu  marching_observers.cuh gpuutils
		$(NVCC) -dc $(NVCCFLAGS) $(CUDA_RUNTIME) -o marching_observers.o  marching_observers.cu
		$(NVCC) -dlink -arch=$(CUDA_COMPUTE_CAPABILITY) -o marching_observers_link.o marching_observers.o -Xcompiler -fPIC

voxelize: voxelize.cu voxelize.cuh gpuutils
		$(NVCC) -dc $(NVCCFLAGS) $(CUDA_RUNTIME) -o voxelize.o voxelize.cu
		$(NVCC) -dlink -arch=$(CUDA_COMPUTE_CAPABILITY) -o voxelize_link.so voxelize.o -Xcompiler -fPIC


all_actions: voxelize marching_observers
		$(CC) $(PYFLAGS) -shared -fPIC -o all_actions.so actions_py.cpp marching_observers_link.o gpuutils.o marching_observers.o voxelize_link.so voxelize.o $(CUDA_RUNTIME)
		# python3 /MieT5/Nearl/src/test_actions.py

test_gpu: testgpu_template.cu 
		$(NVCC) $(NVCCFLAGS) -o a.out testgpu_template.cu 

clean: 
		rm -f *.o *.so






