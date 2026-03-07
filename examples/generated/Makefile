# Auto-generated Makefile for split-feature scoring library
#
# Targets:
#   make cpu          — build CPU main + benchmark
#   make cuda         — build CUDA main + benchmark (requires nvcc)
#   make bench_cpu    — build & run CPU benchmark
#   make bench_cuda   — build & run CUDA benchmark
#   make all          — build everything available
#   make clean        — remove build artifacts
#
# Override compiler:  make CC=gcc-13 cpu
# Override flags:     make CFLAGS="-O3 -march=native" cpu

CC        ?= cc
NVCC      ?= nvcc
CFLAGS    ?= -O2 -Wall
NVFLAGS   ?= -O2
BUILD_DIR ?= ../build

# Detect nvcc
HAS_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

.PHONY: all cpu cuda bench_cpu bench_cuda clean

all: cpu
	if [ -n "$(HAS_NVCC)" ]; then $(MAKE) cuda; fi

# ---- CPU targets ----

cpu: $(BUILD_DIR)/scoring_split_cpu $(BUILD_DIR)/scoring_split_bench_cpu

$(BUILD_DIR)/scoring_split_cpu: scoring_split_main.c scoring_split_core.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ scoring_split_main.c -lm
	@echo "Built: $@"

$(BUILD_DIR)/scoring_split_bench_cpu: scoring_split_bench.c scoring_split_core.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ scoring_split_bench.c -lm
	@echo "Built: $@"

bench_cpu: $(BUILD_DIR)/scoring_split_bench_cpu
	$(BUILD_DIR)/scoring_split_bench_cpu $(BENCH_ARGS)

# ---- CUDA targets ----

cuda: $(BUILD_DIR)/scoring_split_cuda $(BUILD_DIR)/scoring_split_bench_cuda

$(BUILD_DIR)/scoring_split_cuda: scoring_split_main.cu scoring_split_core.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) -o $@ scoring_split_main.cu
	@echo "Built: $@"

$(BUILD_DIR)/scoring_split_bench_cuda: scoring_split_bench.cu scoring_split_core.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) -o $@ scoring_split_bench.cu
	@echo "Built: $@"

bench_cuda: $(BUILD_DIR)/scoring_split_bench_cuda
	$(BUILD_DIR)/scoring_split_bench_cuda $(BENCH_ARGS)

clean:
	rm -f $(BUILD_DIR)/scoring_split_cpu
	rm -f $(BUILD_DIR)/scoring_split_cuda
	rm -f $(BUILD_DIR)/scoring_split_bench_cpu
	rm -f $(BUILD_DIR)/scoring_split_bench_cuda
