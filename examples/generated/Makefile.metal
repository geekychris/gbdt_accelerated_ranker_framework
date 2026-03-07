# Auto-generated Makefile for Metal (Apple Silicon GPU) split-feature scoring
#
# Requires: macOS with Xcode command-line tools (xcrun, clang++)
#           Apple Silicon Mac (M1/M2/M3/M4) for best performance
#
# Targets:
#   make -f Makefile.metal bench     — build & run Metal benchmark
#   make -f Makefile.metal main      — build Metal main driver
#   make -f Makefile.metal metallib  — precompile .metal → .metallib
#   make -f Makefile.metal clean
#
# Override build dir:  make -f Makefile.metal BUILD_DIR=./out bench

CXX       ?= clang++
CXXFLAGS  ?= -O2 -std=c++17 -fobjc-arc
FRAMEWORKS = -framework Metal -framework Foundation
BUILD_DIR ?= ../build

# Detect xcrun (required for Metal shader precompilation)
HAS_XCRUN := $(shell command -v xcrun 2>/dev/null)

.PHONY: all bench main metallib clean

all: bench main

# ---- Main driver ----

main: $(BUILD_DIR)/scoring_split_metal

$(BUILD_DIR)/scoring_split_metal: scoring_split_main_metal.mm scoring_split_metal_core.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o $@ scoring_split_main_metal.mm
	@echo "Built: $@"

# ---- Benchmark ----

bench: $(BUILD_DIR)/scoring_split_bench_metal
	$(BUILD_DIR)/scoring_split_bench_metal $(BENCH_ARGS)

$(BUILD_DIR)/scoring_split_bench_metal: scoring_split_bench_metal.mm scoring_split_metal_core.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o $@ scoring_split_bench_metal.mm
	@echo "Built: $@"

# ---- Precompile Metal shader (optional — for Xcode / production use) ----

metallib: $(BUILD_DIR)/scoring_split_core.metallib

$(BUILD_DIR)/scoring_split_core.metallib: scoring_split_core.metal
	@if [ -z "$(HAS_XCRUN)" ]; then echo "ERROR: xcrun not found"; exit 1; fi
	@mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c scoring_split_core.metal -o $(BUILD_DIR)/scoring_split_core.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/scoring_split_core.air -o $@
	@echo "Built: $@"

clean:
	rm -f $(BUILD_DIR)/scoring_split_metal
	rm -f $(BUILD_DIR)/scoring_split_bench_metal
	rm -f $(BUILD_DIR)/scoring_split_core.air
	rm -f $(BUILD_DIR)/scoring_split_core.metallib
