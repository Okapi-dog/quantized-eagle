#このスクリプトは、異なる量子化設定を使用した線形層のパフォーマンスをベンチマークします。本当にW8A8がこの実機で早くなるのか検証する用。
import torch
import time
import copy
import pandas as pd # 結果を見やすくするために使用
from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig
from torchao.dtypes import SemiSparseLayout
from torchao.quantization.quantize_.workflows import Int4PackingFormat
from torch._inductor.utils import do_bench_using_profiling
from typing import Callable

def benchmark_fn(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3

# --- 設定エリア ---
device = "cuda"
dtype = torch.bfloat16
K, N = 4096, 16384  # 大きめの層 (L2キャッシュ溢れを想定)
M = 1               # Decode想定 (Token length)

# 計測したいバッチサイズのリスト
# A100では 1~128 くらいで挙動が激変します
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

# -----------------

print(f"Device: {device}, Base Shape: [Batch, {M}, {K}] -> [{K}, {N}]")

# 1. モデルの準備 (共通)
# 毎回作り直すとコンパイル時間がかかるため、モデルは先に作ってコンパイルしておく
linear_base = torch.nn.Linear(K, N, bias=False).to(device).to(dtype)

# (A) BF16 Baseline
model_bf16 = copy.deepcopy(linear_base)

# (B) W4A16 (Int4 Weight Only)
model_int4 = copy.deepcopy(linear_base)
config_int4 = Int4WeightOnlyConfig(int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
quantize_(model_int4, config_int4)

# (C) W8A8 (Dynamic Act / Int8 Weight / SemiSparse)
# ※注意: SemiSparseLayoutは本来2:4スパースな重みを前提としますが、ここでは速度計測用として実行します
model_w8a8 = copy.deepcopy(linear_base)
config_w8a8 = Int8DynamicActivationInt8WeightConfig(
    layout=SemiSparseLayout()
)
quantize_(model_w8a8, config_w8a8)

# 2. コンパイル (Max-Autotune)
print("Compiling models... (This takes time on the first run)")
model_bf16 = torch.compile(model_bf16, mode="max-autotune")
model_int4 = torch.compile(model_int4, mode="max-autotune")
model_w8a8 = torch.compile(model_w8a8, mode="max-autotune")

# --- ベンチマーク関数 ---
@torch.no_grad()
def run_benchmark(model, batch_size):
    # 入力生成
    x = torch.randn(batch_size, M, K, device=device, dtype=dtype)
    
    return benchmark_fn(model, x)

# --- 実行ループ ---
results = []

print(f"{'Batch':<6} | {'BF16 (ms)':<12} | {'W4A16 (ms)':<12} | {'W8A8 (ms)':<12} | {'Fastest'}")
print("-" * 65)

for b in BATCH_SIZES:
    try:
        t_bf16 = run_benchmark(model_bf16, b)
        t_int4 = run_benchmark(model_int4, b)
        t_w8a8 = run_benchmark(model_w8a8, b)
        
        # 最速判定
        times = {"BF16": t_bf16, "W4A16": t_int4, "W8A8": t_w8a8}
        fastest = min(times, key=times.get)
        
        print(f"{b:<6} | {t_bf16:<12.4f} | {t_int4:<12.4f} | {t_w8a8:<12.4f} | {fastest}")
        
        results.append({
            "Batch Size": b,
            "BF16 (ms)": t_bf16,
            "W4A16 (ms)": t_int4,
            "W8A8 (ms)": t_w8a8,
            "Fastest": fastest
        })
    except Exception as e:
        print(f"Error at Batch {b}: {e}")

print("-" * 65)
print("Done.")