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


# Check if PyFlags are needed or not
# run the gpuutils and then compile the marching_observers

marching_observers: marching_observers.cu  marching_observers.cuh 
		$(NVCC) -dc $(NVCCFLAGS) -o marching_observers.o  marching_observers.cu
		$(NVCC) -dlink -arch=$(CUDA_COMPUTE_CAPABILITY) -o marching_observers_link.o marching_observers.o -Xcompiler -fPIC



voxelize.so: voxelize.cu voxelize.cuh
		$(NVCC) -dc $(NVCCFLAGS) -o voxelize.o voxelize.cu
		$(NVCC) -dlink -arch=$(CUDA_COMPUTE_CAPABILITY) -o voxelize_link.so voxelize.o -Xcompiler -fPIC


all_actions: voxelize.so marching_observers
		$(CC) $(PYFLAGS) -shared -fPIC -o all_actions.so actions_py.cpp marching_observers_link.o marching_observers.o voxelize_link.so voxelize.o $(CUDA_RUNTIME)
		# python3 /MieT5/BetaPose/src/test_actions.py

test_gpu: testgpu_template.cu 
		$(NVCC) $(NVCCFLAGS) -o a.out testgpu_template.cu 


clean:
		rm -f *.o *.so






