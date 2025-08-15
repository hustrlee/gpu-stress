
# 简单的 GPU 压力测试

一个用于快速压测 NVIDIA GPU（基于 PyTorch）的轻量级脚本，主要通过大矩阵乘法和可选的 2D 卷积保持 GPU 高利用率，便于评估吞吐、显存压力与稳定性。

> 中文 · English

查看英文版请参见 [README.md](README.md)。

## 功能
- 对指定 GPU 连续运行矩阵乘（GEMM）并可选叠加卷积负载。
- 支持多 CUDA streams 并行、不同数值精度（fp16/bf16/fp32）和显存预占。
- 打印估算的 GEMM 吞吐（TFLOPS）和运行统计。

## 依赖
- Python 3.8+
- PyTorch（必须是带 CUDA 的构建以使用 GPU）

推荐在本仓库的 `requirements.txt` 中查看建议版本，并按官方说明安装与本机 CUDA 版本匹配的 PyTorch wheel。

## 使用方法
在有合适 PyTorch CUDA 环境下，运行：

```bash
python gpu_stress.py [--seconds SECONDS] [--size N] [--dtype {fp16,bf16,fp32}] \\
    [--device DEVICE] [--conv] [--streams K] [--reserve-mem-gb G]
```

示例：在 `cuda:0` 上用 fp16、8k 方阵运行 5 分钟并启用卷积：

```bash
python gpu_stress.py --seconds 300 --size 8192 --dtype fp16 --conv
```

## 参数说明
- `--seconds`：压测运行时长（秒），默认 300。
- `--size`：方阵维度 N（用于 N x N 矩阵乘），默认 8192。增大该值会显著提高 FLOPs 和显存占用。
- `--dtype`：计算精度，支持 `fp16`、`bf16`、`fp32`，默认 `fp16`。注意 BF16 需要硬件/驱动支持。
- `--device`：PyTorch 设备字符串，例如 `cuda:0`，默认 `cuda:0`。
- `--conv`：启用额外的大尺寸 2D 卷积以增加计算密度和显存压力（布尔开关）。
- `--streams`：并行 CUDA streams 数量（整数，默认 1）。设置 >1 可尝试提高硬件利用但也可能引入竞争/延迟。
- `--reserve-mem-gb`：预先在 GPU 上分配指定 GB 数的显存以人为增加显存压力（浮点数，默认 0.0）。

## 输出
脚本完成后会打印运行时长、迭代次数与估算的 GEMM 吞吐（单位 TFLOPS），并包含使用的 dtype、方阵大小、streams 数及是否启用了卷积。

## 注意事项与建议
- 请确保安装的 PyTorch 版本与主机上的 CUDA 驱动兼容，否则 `torch.cuda.is_available()` 可能为 False。
- 大尺寸（例如 N=8192）和启用卷积会消耗大量显存，请先用小参数调试以避免 OOM。
- 如果想精确控制 PyTorch 的 CUDA wheel，请参考官方安装页 https://pytorch.org，选择与你的 CUDA 版本匹配的 wheel。

## 许可证
见仓库根目录 `LICENSE`。
