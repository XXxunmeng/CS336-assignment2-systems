from cs336_basics.transformer import TransformerLM
from cs336_basics.train import cross_entropy
from cs336_basics.train import AdamW
import argparse
import timeit
import torch
import statistics
from cs336_basics.train import get_batch
import numpy as np
import torch.cuda.nvtx as nvtx
import cs336_basics.transformer as transformer
from contextlib import nullcontext

MODEL_SPECS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "20B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36}
}

def autocast_ctx(args, device):
    if args.dtype == "bfloat16" and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", type=str, default="forward_backward", choices=["forward", "forward_backward", "full_step"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--steps", type=int, default=10)
    # parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def benchmark(model, optimizer, input_ids, target_ids, args):
    device = args.device
    model.to(device)
    model.train()
    for _ in range(args.warmup):
        with nvtx.range("warmup"), autocast_ctx(args, device):
            logits = model(input_ids)
            loss = cross_entropy(logits, target_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if args.device == "cuda":
        torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    optimizer_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(args.steps):
        start_event.record()
        with nvtx.range("forward"), autocast_ctx(args, device):
            logits = model(input_ids)
        end_event.record()
        if args.device == "cuda":
            torch.cuda.synchronize()
        forward_times.append((start_event.elapsed_time(end_event)))

        start_event.record()
        with nvtx.range("backward"):
            with autocast_ctx(args, device):
                loss = cross_entropy(logits, target_ids)
            loss.backward()
        end_event.record()
        if args.device == "cuda":
            torch.cuda.synchronize()
        backward_times.append((start_event.elapsed_time(end_event)))

        start_event.record()
        with nvtx.range("optimizer"):
            optimizer.step()
            optimizer.zero_grad()
        end_event.record()
        if args.device == "cuda":
            torch.cuda.synchronize()
        optimizer_times.append((start_event.elapsed_time(end_event)))

    print(f"forward time: {sum(forward_times) / len(forward_times)} ms, std: {statistics.stdev(forward_times)}")
    print(f"backward time: {sum(backward_times) / len(backward_times)} ms, std: {statistics.stdev(backward_times)}")
    print(f"optimizer time: {sum(optimizer_times) / len(optimizer_times)} ms, std: {statistics.stdev(optimizer_times)}")

def scaled_dot_product_attention_nvtx(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    with nvtx.range("attention_scores"):
        d_k = query.shape[-1]
        score = query @ key.transpose(-2, -1) / d_k ** 0.5
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
    with nvtx.range("attention_softmax"):
        score = transformer.softmax(score, dim=-1)
    with nvtx.range("attention_matmul"):
        result = score @ value

    return result

def attention_nvtx():
    transformer.scaled_dot_product_attention = scaled_dot_product_attention_nvtx

def main():
    args = parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device

    attention_nvtx()

    # # 打印本次实验配置，方便写报告和重定向到 CSV
    # print(
    #     f"[config] model_size={args.model_size} vocab={args.vocab_size} "
    #     f"batch={args.batch_size} ctx={args.context_length} "
    #     f"mode={args.mode} dtype={args.dtype} "
    #     f"warmup={args.warmup} measure={args.measure} device={device}"
    # )

    spec = MODEL_SPECS[args.model_size]
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=spec["d_model"],
        num_layers=spec["num_layers"],
        num_heads=spec["num_heads"],
        d_ff=spec["d_ff"],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    dataset = np.load("../../hw1/CS336-assignment1-basics/data/TinyStoriesV2-GPT4-train_ids.npy")
    input_ids, target_ids = get_batch(dataset, args.batch_size, args.context_length, device)

    # 4. 跑 benchmark
    benchmark(model, optimizer, input_ids, target_ids, args)


if __name__ == "__main__":
    main()

    #uv run cs336_systems/benchmark.py --model-size small --context-length 256
    # nsys profile -o results/nsys/small_ctx512 uv run cs336_systems/benchmark.py --model-size small --context-length 512 --warmup 10 --steps 100