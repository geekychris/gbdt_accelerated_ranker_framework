# GBDT Accelerated Ranker Framework

A framework for compiling trained XGBoost Gradient Boosted Decision Tree (GBDT) models into hardware-accelerated ranking kernels — targeting NVIDIA CUDA GPUs, Apple Silicon Metal GPUs, and single-threaded C for portability.

The core use case is **ad / document ranking**: a user feature vector is combined with a large set of candidate document vectors, and the GBDT model scores every `[user ‖ document]` pair in parallel on the GPU.  This pattern appears in recommendation systems, ad auctions, and search ranking.

---

## Architecture

```
XGBoost Model (.ubj / .json)
        │
        ▼
  cuda_codegen (Python)
        │
        ├── scoring_split_core.cuh   ← CUDA header-only library
        ├── scoring_split_core.h     ← CPU header-only library
        └── scoring_split_core.metal ← Metal shader (Apple Silicon)
        │
        ├── nvcc  ──► CUDA binary   (NVIDIA GPU, ~50-300M items/sec)
        ├── clang ──► Metal binary  (Apple M-series, ~350M items/sec)
        └── gcc   ──► C binary      (CPU baseline, ~6M items/sec)
        │
        ▼
  runtime/  (scorched — Rust gRPC server)
        │
        └── gRPC API: LoadDataset, ScoreTopK, ScoreBatch
```

### Split-feature mode

The key design: item/document features are loaded to GPU memory once (or when the candidate set changes).  Per-request, only the user feature vector is transferred.  The GPU concurrently scores all documents.

```
GPU memory (resident between requests):
  d_items[N × item_features]    ← N candidate documents

Per scoring request (tiny transfer):
  h_user[user_features]         → GPU → score all N documents → top-K result
```

---

## Components

| Directory | Description |
|-----------|-------------|
| `python/cuda_codegen/` | Code generator: XGBoost → C / CUDA / Metal source |
| `examples/train/` | XGBoost training example (fraud scoring dataset) |
| `examples/generated/` | Pre-generated output for the example model |
| `runtime/` | `scorched` — Rust gRPC scoring server wrapping the generated C library |
| `docs/` | Architecture documentation |

---

## Quick start

### 1. Train a model

```bash
cd examples/train
pip install -r ../../python/requirements.txt
python run_gbdt_example.py          # trains + saves model.ubj
```

### 2. Generate scoring code

```bash
cd python

# CPU + CUDA library (split: 5 user features, 10 document features)
python -m cuda_codegen generate \
  --model ../examples/train/models/model.ubj \
  --metadata ../examples/train/models/training_result.json \
  --output ../examples/generated \
  --user-features 5 --library

# Apple Silicon Metal library
python -m cuda_codegen generate \
  --model ../examples/train/models/model.ubj \
  --metadata ../examples/train/models/training_result.json \
  --output ../examples/generated \
  --user-features 5 --library --metal
```

### 3. Build and benchmark

```bash
cd examples/generated

# CPU benchmark
make bench_cpu

# CUDA benchmark (requires NVIDIA GPU + nvcc)
make bench_cuda

# Metal benchmark (requires Apple Silicon Mac)
make -f Makefile.metal bench
```

### 4. Run the gRPC scoring server

```bash
cd runtime
./scripts/build.sh --release
./target/release/scorched --port 50051
```

---

## Benchmark results (example model: 500 trees, 15 features, binary:logistic)

Platform: Apple M4 Max, split 5 user + 10 document features

| Documents | CPU (1 thread) | Metal GPU | Speedup |
|-----------|----------------|-----------|---------|
| 100K | 15 ms / 6.5M/s | 0.47 ms / 213M/s | **32×** |
| 500K | 76 ms / 6.6M/s | 1.6 ms / 320M/s | **49×** |
| 1M | 154 ms / 6.5M/s | 2.9 ms / 346M/s | **53×** |
| 2M | 307 ms / 6.5M/s | 5.6 ms / 360M/s | **55×** |
| 5M | 765 ms / 6.5M/s | 13 ms / 376M/s | **57×** |

Top-100 ranking (fused score + select):

| Documents | CPU (score+qsort) | Metal GPU | Speedup |
|-----------|-------------------|-----------|---------|
| 1M | 228 ms | 53 ms | **4×** |
| 5M | 1,186 ms | 47 ms | **25×** |

---

## Code generator CLI

```bash
cd python

# Generate CUDA split-feature library
python -m cuda_codegen generate \
  --model MODEL.ubj [--metadata training_result.json] \
  --output OUTPUT_DIR \
  --user-features N \
  --library

# Generate Metal (Apple Silicon) library
python -m cuda_codegen generate ... --library --metal

# Generate CPU-only C library
python -m cuda_codegen generate ... --library --cpu

# Verify generated code against XGBoost predictions
python -m cuda_codegen verify --model MODEL.ubj --n-samples 1000

# Verify split-feature + top-K
python -m cuda_codegen verify --model MODEL.ubj --user-features N --topk 100
```

### Generated file summary

| File | Description |
|------|-------------|
| `scoring_split_core.cuh` | CUDA header-only library (tree functions + GPU kernel + host API) |
| `scoring_split_core.h` | CPU header-only library |
| `scoring_split_core.metal` | Metal shader (Apple Silicon GPU) |
| `scoring_split_metal_core.h` | Obj-C++ header-only Metal library (embeds shader) |
| `scoring_split_bench.cu/c/.mm` | Benchmark drivers |
| `scoring_split_main.cu/c/.mm` | Standalone main() drivers |
| `Makefile` / `Makefile.metal` | Build targets |

---

## Host API

The same C API is exposed by all three backends (CUDA, Metal, CPU):

```c
// Allocate context (pre-allocates GPU memory for up to max_items documents)
ScoringContext* scoring_context_create(int max_items);
void            scoring_context_destroy(ScoringContext* ctx);

// Upload document features to GPU (call once, or when candidates change)
int load_items(ScoringContext* ctx, const float* items, int n_items);

// Score all documents against one user — returns all scores
int score_user(ScoringContext* ctx, const float* user, float* scores);

// Score all documents and return only the top-K (fused kernel, less D→H transfer)
int score_user_topk(ScoringContext* ctx, const float* user, int k,
                    int* topk_indices, float* topk_scores);
```

---

## Design notes

**Inlined decision trees** — Each tree becomes a nested `if/else` function compiled into the GPU instruction stream, not a data structure traversed at runtime. Branch predictors handle the paths. This is optimal for models up to ~1000 trees × depth 8; for larger models a flat-array representation is more cache-friendly.

**NaN handling** — Missing values use IEEE 754 semantics without explicit `isnan()`:
- NaN routes same as "yes" → `!(f[i] >= threshold)` (true for both `f < t` and NaN)
- NaN routes same as "no"  → `f[i] < threshold` (false for NaN, falls to else)

**Apple Silicon unified memory** — `MTLResourceStorageModeShared` buffers are accessible from both CPU and GPU without any memcpy overhead. `load_items()` is a plain `memcpy` at ~17 GB/s; score results are read directly. No PCIe bus involved.

**Top-K algorithm** — Two-phase: (1) each GPU block/threadgroup scores its range and maintains a sorted local top-K in shared memory using insertion sort; (2) the host merges `grid_size × K` block candidates via `qsort`. This avoids transferring all N scores back to the host.

---

## Related

- The `feature_store` repository uses this framework as a companion service for model-based ranking.
