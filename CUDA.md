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

- **`src/torchlogix/layers/dense.py`** — `LogicDenseCudaFunction` (a `torch.autograd.Function`) calls `torchlogix_cuda.forward()`, `.backward_w()`, `.backward_x()`, and `torchlogix_cuda.eval()`.
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

Tested on MNIST (784→10, default architecture):

| Backend | Throughput | Notes |
|---------|-----------|-------|
| CUDA | ~236 it/s | `torchlogix_cuda` extension |
| Python fallback | ~15 it/s | Pure PyTorch (`functional.py`) |

~15× speedup from CUDA kernels on a single GPU.
