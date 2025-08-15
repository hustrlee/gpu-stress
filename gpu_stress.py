#!/usr/bin/env python3
# gpu_stress.py
import time, argparse, math
import torch
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description="GPU stress (matmul/conv) for N seconds")
    parser.add_argument("--seconds", type=int, default=300, help="持续时长（秒）")
    parser.add_argument("--size", type=int, default=8192, help="方阵大小 N（N x N）")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="计算精度")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU 设备")
    parser.add_argument("--conv", action="store_true", help="额外加入卷积负载（增强占用）")
    parser.add_argument("--streams", type=int, default=1, help="并行 CUDA streams 数（>1 可能更吃满）")
    parser.add_argument("--reserve-mem-gb", type=float, default=0.0, help="预占显存（GB），增加显存压力")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要可用的 NVIDIA GPU 和 CUDA 版 PyTorch"
    device = torch.device(args.device)

    # 精度设置
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # 预占显存（可选）
    reserve = None
    if args.reserve_mem_gb > 0:
        elems = int(args.reserve_mem_gb * (1024**3) / torch.tensor([], dtype=dtype).element_size())
        reserve = torch.empty(elems, dtype=dtype, device=device)
        reserve.fill_(1)

    # 创建 streams
    streams = [torch.cuda.Stream(device=device) for _ in range(max(1, args.streams))]

    N = args.size
    # 主负载张量（每个 stream 一套，避免竞争）
    mats = []
    for _ in streams:
        A = torch.randn((N, N), device=device, dtype=dtype)
        B = torch.randn((N, N), device=device, dtype=dtype)
        C = torch.empty((N, N), device=device, dtype=dtype)
        mats.append((A, B, C))

    # 可选卷积负载（较大 2D 卷积）
    conv_shapes = []
    if args.conv:
        # 形状尽量大但不炸显存，可按需调
        C_in, C_out, H, W, K = 64, 128, 2048, 2048, 3
        for _ in streams:
            x = torch.randn((1, C_in, H, W), device=device, dtype=dtype)
            w = torch.randn((C_out, C_in, K, K), device=device, dtype=dtype)
            conv_shapes.append((x, w))

    # 预热
    warmup_iters = 10
    for si, s in enumerate(streams):
        with torch.cuda.stream(s):
            A, B, C = mats[si]
            for _ in range(warmup_iters):
                C = A @ B
                if args.conv:
                    x, w = conv_shapes[si]
                    y = F.conv2d(x, w, stride=1, padding=1)
    torch.cuda.synchronize()

    # 压测
    start = time.time()
    end_time = start + args.seconds
    iters = 0
    matmul_flops = 0.0

    # 每次迭代统计一次 FLOPs：2*N^3（标准 GEMM）
    flops_per_matmul = 2.0 * (N ** 3)

    while True:
        now = time.time()
        if now >= end_time:
            break

        for si, s in enumerate(streams):
            with torch.cuda.stream(s):
                A, B, C = mats[si]
                # 主要计算：矩阵乘
                C = A @ B
                # 再来一些逐元素操作增加指令吞吐
                C = torch.sin(C) + torch.cos(C) * 1.0001

                if args.conv:
                    x, w = conv_shapes[si]
                    y = F.conv2d(x, w, stride=1, padding=1)
                    y = F.relu(y)

        iters += 1
        matmul_flops += flops_per_matmul * len(streams)

        # 周期性同步，防止过度排队导致温度读数滞后
        if iters % 20 == 0:
            torch.cuda.synchronize()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    tflops = (matmul_flops / elapsed) / 1e12
    print(f"运行 {elapsed:.1f}s, 迭代 {iters} 次, 估算 GEMM 吞吐 ≈ {tflops:.2f} TFLOPS "
          f"(dtype={args.dtype}, N={N}, streams={len(streams)}, conv={args.conv})")

if __name__ == "__main__":
    main()

