# Simple GPU Stress Test
A lightweight script for quickly stressing NVIDIA GPUs (PyTorch-based). It uses large matrix multiplications (GEMM) and an optional large 2D convolution to keep the GPU busy, making it useful for evaluating throughput, memory pressure, and stability.

> English · 中文

For the Chinese README, see [README_cn.md](README_cn.md).

## Features
- Supports multiple CUDA streams, selectable numeric precisions (fp16 / bf16 / fp32), and optional reserved GPU memory.
- Continuously runs matrix multiplications on a chosen GPU and optionally adds a convolution workload.
- Prints estimated GEMM throughput (TFLOPS) and run statistics.

## Requirements
- Python 3.8+
- PyTorch with CUDA support (a CUDA-enabled build is required to use the GPU)

See requirements.txt for recommended versions. Install a PyTorch wheel that matches your system CUDA version from the official instructions.

## Usage
With a compatible PyTorch + CUDA environment, run:

```bash
python gpu_stress.py [--seconds SECONDS] [--size N] [--dtype {fp16,bf16,fp32}] \
    [--device DEVICE] [--conv] [--streams K] [--reserve-mem-gb G]
```

Example: run 5 minutes on `cuda:0` with fp16 and an 8k square matrix, enabling convolution:

```bash
python gpu_stress.py --seconds 300 --size 8192 --dtype fp16 --conv
```

## Command-line options
- `--seconds`: Duration of the stress test in seconds (default: 300).
- `--size`: Square matrix dimension N (for N x N matrix multiply), default 8192. Increasing N raises FLOPs and memory usage significantly.
- `--dtype`: Compute precision: `fp16`, `bf16`, or `fp32` (default: `fp16`). BF16 requires hardware and driver support.
- `--device`: PyTorch device string, e.g. `cuda:0` (default: `cuda:0`).
- `--conv`: Enable additional large 2D convolution to increase compute density and memory pressure (flag).
- `--streams`: Number of parallel CUDA streams (integer, default: 1). Values >1 may increase utilization but can also introduce contention/delay.
- `--reserve-mem-gb`: Pre-allocate this many GB of GPU memory to artificially increase memory pressure (float, default: 0.0).

## Output
After the run, the script prints runtime, iteration count, and an estimated GEMM throughput in TFLOPS, plus the used dtype, matrix size, stream count, and whether convolution was enabled.

## Notes & Recommendations
- Ensure the installed PyTorch wheel is compatible with your host CUDA drivers; otherwise `torch.cuda.is_available()` may be False.
- Large sizes (e.g., N=8192) and enabling convolution consume lots of GPU memory—test with smaller settings first to avoid OOM.
- For precise control over the PyTorch CUDA wheel, consult the official install page at https://pytorch.org and pick the wheel matching your CUDA version.

## License
MIT
