# CUDA Extension for torchlogix

This document describes the `torchlogix_cuda` native CUDA extension — its origin, architecture, build process, and troubleshooting.

## Background

torchlogix is a fork of [Felix-Petersen/difflogic](https://github.com/Felix-Petersen/difflogic) (NeurIPS 2022). The upstream project ships a CUDA extension called `difflogic_cuda` that accelerates logic-gate layer forward/backward passes on GPU. When torchlogix was forked and renamed, the CUDA source was not included, causing a `NameError: name 'torchlogix_cuda' is not defined` at runtime.

We ported the upstream CUDA kernels, renaming the module from `difflogic_cuda` to `torchlogix_cuda` and fixing compatibility issues for modern PyTorch (2.x).

## Architecture

### Source Files

| File | Purpose |
|------|---------|
| `src/torchlogix/cuda/torchlogix_kernel.cu` | CUDA kernels (~700 lines) |
| `src/torchlogix/cuda/torchlogix.cpp` | pybind11 bindings exposing 6 functions to Python |
| `setup.py` | Build configuration via `torch.utils.cpp_extension.CUDAExtension` |

### Exposed Functions

The extension exposes 6 functions via `torchlogix_cuda`:

| Function | Mode | Description |
|----------|------|-------------|
| `forward(x, a, b, w)` | Training | Forward pass of differentiable logic gate layer. Dispatched over `float`, `double`, `half`. |
| `backward_w(x, a, b, grad_y)` | Training | Gradient w.r.t. gate-selection weights `w`. Uses atomic floating-point adds with per-warp reduction. |
| `backward_x(x, a, b, w, grad_y, ...)` | Training | Gradient w.r.t. inputs `x`. Uses pre-computed connectivity maps (`given_x_indices_of_y_start`, `given_x_indices_of_y`). |
| `eval(x, a, b, w)` | Inference | Hard (argmax) gate evaluation — selects the gate with the highest weight and evaluates it exactly (no softmax blending). |
| `tensor_packbits_cuda(t, bit_count)` | Compilation | Packs a boolean tensor into packed-bit integers for compiled (inference-only) models. |
| `groupbitsum(b, pad_len, k)` | Compilation | Group-wise bit-population-count sum used by `PackBitsTensor` operations. |

### Kernel Details

**Training kernels** (`forward`, `backward_w`, `backward_x`) use `AT_DISPATCH_FLOATING_TYPES_AND_HALF` to support `float32`, `float64`, and `float16`.

**Inference/utility kernels** (`eval`, `tensor_packbits_cuda`, `groupbitsum`) use `AT_DISPATCH_INTEGRAL_TYPES` since they operate on integer/boolean packed representations.

**Logic gates**: Each neuron implements one of 16 possible binary logic operations (AND, OR, XOR, etc.) selected by a continuous weight vector `w` of length 16. During training the output is a soft-weighted combination of all 16 gates; at eval time the argmax gate is selected.

**Backward w** is the most complex kernel: it uses `BACKWARD_W_BATCH_THREADS=32` threads per gate and performs an intra-warp reduction via `__shfl_down_sync` before an atomic add to the gradient tensor. A custom `AtomicFPOp<at::Half>` struct handles half-precision atomics via CAS loops.

### Consumer Code

- **`src/torchlogix/layers/dense.py`** — `LogicDenseCudaFunction` (a `torch.autograd.Function`) calls `torchlogix_cuda.forward()`, `.backward_w()`, `.backward_x()`, and `torchlogix_cuda.eval()`. The native CUDA kernel only supports `parametrization='raw'` (16-gate blend); when `parametrization='walsh'` (or `'anf'`) is used, the dense layer automatically falls back to the Python path.
- **`src/torchlogix/layers/conv.py`** — `LogicConv2d` uses the flatten-and-reuse strategy to call `LogicDenseCudaFunction` for `parametrization='raw'`, and a flat-index GPU-accelerated PyTorch path for `parametrization='walsh'`.
- **`src/torchlogix/packbitstensor.py`** — uses `torchlogix_cuda.tensor_packbits_cuda()` and `torchlogix_cuda.groupbitsum()` for model compilation.

Both files import via a guarded try/except:
```python
try:
    import torchlogix_cuda
except ImportError:
    warnings.warn("failed to import torchlogix_cuda. ...")
```

## Build Prerequisites

| Requirement | Tested Version |
|-------------|---------------|
| Python | 3.12 |
| PyTorch | 2.10.0 |
| CUDA Runtime | 12.8 |
| nvcc (CUDA compiler) | 12.8 |
| GCC (host compiler) | Provided by conda / system |

The nvcc version **must** match the CUDA version PyTorch was built with. Check with:
```bash
python -c "import torch; print(torch.version.cuda)"
```

## Build Instructions

### 1. Install nvcc

If `nvcc` is not available on the system (common on shared clusters), install it into your conda environment:

```bash
conda install -n <env_name> -y -c nvidia cuda-nvcc=12.8
```

Replace `12.8` with your PyTorch CUDA version.

### 2. Set Environment Variables

```bash
# Point CUDA_HOME to the conda environment (which now contains nvcc)
export CUDA_HOME=$CONDA_PREFIX

# Ensure the conda libstdc++ is found (required if conda GCC is newer than system GCC)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### 3. Build and Install

```bash
pip install --no-build-isolation -e .
```

**`--no-build-isolation`** is required because `setup.py` imports `torch` at the top level (`from torch.utils.cpp_extension import BuildExtension, CUDAExtension`). Without this flag, pip creates an isolated build environment that does not have PyTorch installed.

### 4. Verify

```python
import torchlogix_cuda
print(dir(torchlogix_cuda))
# ['__doc__', '__file__', ..., 'backward_w', 'backward_x', 'eval', 'forward', 'groupbitsum', 'tensor_packbits_cuda']
```

## Changes from Upstream difflogic

### Rename

All references to `difflogic_cuda` were renamed to `torchlogix_cuda`:
- Module name in `PYBIND11_MODULE`
- Extension name in `setup.py`
- Import statements in Python code

### PyTorch 2.x Compatibility: `.type()` → `.scalar_type()`

The upstream code used the deprecated `.type()` method inside `AT_DISPATCH_*` macros:

```cpp
// Old (upstream) — causes compile error on PyTorch ≥ 2.x
AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "forward", ([&] { ... }));
```

PyTorch 2.x removed the `.type()` accessor and requires `.scalar_type()` instead:

```cpp
// New (fixed)
AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "forward", ([&] { ... }));
```

This change was applied in **6 locations** across `torchlogix_kernel.cu` — 3 in the floating-point dispatch macros and 3 in the integral dispatch macros.

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'` during build

`setup.py` does `from torch.utils.cpp_extension import ...` at the top level. Use `--no-build-isolation` so pip uses the active environment instead of an isolated one:

```bash
pip install --no-build-isolation -e .
```

### `CUDA_HOME environment variable is not set`

Set it to wherever nvcc lives. If using conda-installed nvcc:

```bash
export CUDA_HOME=$CONDA_PREFIX
```

### `nvcc not found`

Install nvcc into your conda environment:

```bash
conda install -n <env_name> -y -c nvidia cuda-nvcc=12.8
```

### `.type()` compilation errors

```
error: 'at::DeprecatedTypeProperties' has no member named 'is_cuda'
```

Replace `.type()` with `.scalar_type()` in all `AT_DISPATCH_*` macros and replace `x.type().is_cuda()` with `x.is_cuda()` in `CHECK_CUDA`.

### `CXXABI_1.3.15 not found` at import time

This means the compiled extension links against a newer libstdc++ (from conda GCC) than the system provides at runtime. Fix by prepending the conda lib directory:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

For a permanent fix, add this to your shell profile or conda activation script:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' \
    > $CONDA_PREFIX/etc/conda/activate.d/libstdcxx.sh
```

## Performance

### Dense Layers (`LogicDense`)

Tested on MNIST (784→10, default architecture):

| Backend | Throughput | Notes |
|---------|-----------|-------|
| CUDA | ~236 it/s | `torchlogix_cuda` extension |
| Python fallback | ~15 it/s | Pure PyTorch (`functional.py`) |

~15× speedup from CUDA kernels on a single GPU.

### Convolutional Layers (`LogicConv2d`)

Tested on MNIST-like first conv layer (28×28 input, K=16 kernels, RF=5, tree_depth=3, batch=64):

| Parametrization | Backend | Throughput | Speedup |
|-----------------|---------|------------|---------|
| `raw` | CUDA (native kernel) | ~435 it/s (2.3 ms/iter) | **11.6×** |
| `raw` | Python (on GPU) | ~37 it/s (26.7 ms/iter) | baseline |
| `walsh` | CUDA (flat-index) | ~75 it/s (13.4 ms/iter) | **1.5×** |
| `walsh` | Python (on GPU) | ~50 it/s (20.0 ms/iter) | baseline |

The `raw` parametrization benefits dramatically because the native CUDA kernel fuses the 16-gate blend into a single kernel launch. The `walsh` parametrization sees a modest speedup from the flat 1-D index gather being more efficient than multi-dimensional fancy indexing, but the Walsh basis computation itself (4-coefficient dot product + sigmoid) runs the same PyTorch ops in both paths.


---

## CUDA Acceleration for Convolutional Logic Layers (`LogicConv2d`)

### Problem

The existing CUDA extension only accelerates `LogicDense` layers. `LogicConv2d` (and `LogicConv3d`) fall back to pure-Python/PyTorch, creating a bottleneck in convolutional logic gate networks.

### Key Structural Differences

| Aspect | LogicDense | LogicConv2d |
|--------|-----------|-------------|
| **Input** | `(batch, in_dim)` — flat 2-D | `(batch, C, H, W)` — 4-D |
| **Connections** | Flat 1-D index pairs `a[col]`, `b[col]` | Multi-dim sliding-window indices `(h, w, c)` per kernel × spatial position × sample |
| **Weights** | Single `(out_dim, 16)` tensor | A tree of weight lists: `tree_weights[level][node]` each `(num_kernels, 16)`, with `2^0 + 2^1 + … + 2^depth` nodes total |
| **Forward** | Single gather + 16-gate blend | Multi-level binary tree: level 0 gathers from 4-D input, levels 1–depth pair-wise reduce |
| **Weight sharing** | None — each neuron has its own weights | Same kernel weights applied at *every* spatial position |

### Evaluated Approaches

Three approaches were considered for bringing CUDA acceleration to the conv layers:

#### Approach A: Flatten-and-Reuse Dense Kernel (Implemented)

**Idea:** Flatten the 4-D conv input to 2-D, convert multi-dimensional sliding-window indices to linear 1-D offsets, replicate the shared weights across spatial positions, and call the *existing* `LogicDenseCudaFunction` at each tree level.

**How it works:**

1. **Level 0 — flatten sliding-window indices:**
   - Input `(batch, C, H_p, W_p)` → reshape to `(C·H_p·W_p, batch)`.
   - Each multi-dim index triple `(h, w, c)` is converted to a flat offset: `c·H_p·W_p + h·W_p + w`.
   - The total "virtual neurons" = `K × S × D` where K = kernels, S = spatial positions, D = 2^tree_depth.

2. **Weight replication:**
   - Each tree node has weights `(K, 16)` shared across all S spatial positions.
   - Expand: `(D, K, 16)` → `(K, S, D, 16)` → reshape to `(K·S·D, 16)`.
   - Autograd automatically sums gradients back over the S dimension.

3. **Tree reduction (levels 1+):**
   - Intermediate result `(batch, K, S, D_prev)` → flatten to `(K·S·D_prev, batch)`.
   - Pair-wise indices: for output neuron `(k, s, d_new)`, read `(k, s, 2·d_new)` and `(k, s, 2·d_new + 1)`.
   - Repeat until D = 1, then reshape to `(batch, K, H_out, W_out)`.

4. **Reverse maps** (`given_x_indices_of_y_start`, `given_x_indices_of_y`) are precomputed at init time for the CUDA backward_x kernel.

| Property | Value |
|----------|-------|
| **Effort** | Low — Python-only changes, no new CUDA code |
| **Speedup** | ~11.6× for `raw` (native CUDA kernel); ~1.5× for `walsh` (flat-index gather) |
| **Risk** | Low — proven kernel, all PyTorch autograd |
| **Limitations** | Weight replication increases memory |

**Memory estimate** for MNIST CNN (28×28, 3×3 RF, stride 1, 16 kernels, tree_depth=3):
- Virtual neurons at level 0: 16 × 676 × 8 = 86,528
- Replicated weight tensor: 86,528 × 16 × 4 bytes ≈ 5.3 MB (float32) — very manageable.

**Status: Implemented** in `src/torchlogix/layers/conv.py`. Supports both `parametrization='raw'` and `parametrization='walsh'`. Set `implementation='cuda'` or leave as `None` (auto-selects CUDA when `device='cuda'`).

#### Approach B: Custom Conv CUDA Kernel (Level 0 Only)

**Idea:** Write a new CUDA kernel for the level-0 gather that natively handles 4-D indexing `(h, w, c)` and shared-weight gradient accumulation. Keep levels 1+ in PyTorch (they're small).

**Advantages over A:**
- Avoids the memory overhead of weight replication across spatial positions.
- Eliminates the flatten/reshape overhead between levels.
- The backward_w kernel could efficiently sum gradients across spatial positions using shared memory or warp-level reduction instead of relying on autograd's scatter.

**Required work:**
- New forward kernel parameterized by `(K, S, D)` with 4-D index lookups.
- New backward_w kernel with spatial-position reduction (similar to the existing per-warp reduction, but with an extra loop over S).
- New backward_x kernel with a reverse-map structure adapted for 4-D indices.
- Pybind11 bindings and integration into `conv.py`.

| Property | Value |
|----------|-------|
| **Effort** | Medium — ~500 lines of new CUDA code |
| **Speedup** | Slightly better than A (avoids flatten overhead + replicated weight memory) |
| **Risk** | Medium — new kernel code to write, test, and maintain across dtypes/GPU architectures |

**Status: Not implemented.** Recommended only if profiling shows Approach A's weight replication or reshape overhead is a bottleneck.

#### Approach C: Fully Custom Conv CUDA Kernels (All Levels)

**Idea:** Write custom CUDA kernels for *every* tree level (both the sliding-window level 0 and the pair-wise reduction levels 1+), potentially fusing multiple levels into a single kernel launch.

**Advantages over B:**
- Eliminates all intermediate tensor allocations and Python-level loops.
- Could fuse the tree reduction into a single kernel with shared-memory staging.

**Additional complexity:**
- The reduction levels are geometrically shrinking (D/2, D/4, …, 1), so the total work in levels 1+ is comparable to a single level-0 pass. Fusing provides diminishing returns.
- Much more complex kernel with variable-depth loops and synchronization.

| Property | Value |
|----------|-------|
| **Effort** | High — ~1000+ lines of CUDA, complex fused backward |
| **Speedup** | Marginal over B (tree reduction levels are already fast in PyTorch) |
| **Risk** | High — complex kernel, extensive testing required |

**Status: Not implemented.** The extra speedup is unlikely to justify the engineering cost.

### Implementation Details (Approach A)

#### Files Modified

- **`src/torchlogix/layers/conv.py`** — Added CUDA path to `LogicConv2d`:
  - `_build_reverse_map()`: Builds CSR-like reverse connectivity maps for backward_x.
  - `_prepare_cuda_indices()`: Precomputes flat 1-D indices and reverse maps at init time.
  - `_get_weighting_func()`: Returns the appropriate weight transform for the `forward_sampling` mode.
  - `forward_cuda()`: Dispatches to `_forward_cuda_raw()` or `_forward_cuda_walsh()` based on parametrization.
  - `_forward_cuda_raw()`: Uses the native `LogicDenseCudaFunction` kernel with replicated weights.
  - `_forward_cuda_walsh()`: Uses precomputed flat indices for 1-D gather, then Walsh basis expansion + activation via PyTorch ops on GPU.

- **`src/torchlogix/layers/dense.py`** — Added automatic fallback: when `implementation='cuda'` but `parametrization` is not `'raw'`, the dense layer silently falls back to the Python path. This prevents the CUDA kernel (which returns 16-column weight gradients) from crashing on Walsh weights (which have only 4 columns).

#### Usage

```python
# Automatic: selects CUDA when device='cuda' (any parametrization)
layer = LogicConv2d(in_dim=28, channels=1, num_kernels=16,
                    tree_depth=3, receptive_field_size=5,
                    device='cuda')

# Works with both parametrizations:
layer_raw = LogicConv2d(..., device='cuda', parametrization='raw')     # native CUDA kernel
layer_walsh = LogicConv2d(..., device='cuda', parametrization='walsh') # flat-index GPU path

# Explicit:
layer = LogicConv2d(..., device='cuda', implementation='cuda')

# Force Python path on GPU (for debugging):
layer = LogicConv2d(..., device='cuda', implementation='python')
```

#### Limitations

- **`LogicDense` with Walsh uses Python fallback.** The native CUDA kernel only supports the 16-gate `raw` blend. When a `LogicDense` layer has `parametrization='walsh'`, it automatically falls back to the Python path (still runs on GPU via PyTorch). This is transparent and requires no user action.
- **`LogicConv3d` not yet accelerated.** The same flatten-and-reuse approach applies identically to 3-D convolutions — the only difference is a 4-D index tuple `(h, w, d, c)` instead of 3-D `(h, w, c)`. This can be added with straightforward changes to `_prepare_cuda_indices`.
- **Memory.** The replicated weight tensor scales as `K × S × D × W × sizeof(float)` where W=16 for raw and W=4 for Walsh. For large spatial dimensions (e.g., 224×224 ImageNet with raw) this can reach ~1.6 GB. Monitor with `torch.cuda.memory_allocated()`.
