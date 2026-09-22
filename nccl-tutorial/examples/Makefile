CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= $(abspath ../../nccl/build)
NVCC ?= $(CUDA_HOME)/bin/nvcc
MPICXX ?= mpicxx
BUILD := build
CPPFLAGS += -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
CXXFLAGS ?= -O2 -std=c++17
LDFLAGS += -L$(NCCL_HOME)/lib -L$(CUDA_HOME)/lib64
LDLIBS += -lnccl -lcudart -pthread

.PHONY: all mpi
all: $(BUILD)/single_process_allreduce
mpi: $(BUILD)/mpi_allreduce

$(BUILD):
	mkdir -p "$@"

$(BUILD)/single_process_allreduce: single_process_allreduce.cu | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(CXXFLAGS) -Xcompiler -pthread $< $(LDFLAGS) -lnccl -lcudart -o $@

$(BUILD)/mpi_allreduce: mpi_allreduce.cu | $(BUILD)
	$(MPICXX) -x c++ $(CPPFLAGS) $(CXXFLAGS) $< -x none $(LDFLAGS) $(LDLIBS) -o $@
