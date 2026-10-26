import os
import torch
import torch.distributed as dist
from cs336_systems.ddp import NaiveDDP, FlatDDP, OverlapDDP
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"]  = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def make_data(rank, world_size, batch=8, seq=64, dim=128, device="cpu"):
    g = torch.Generator().manual_seed(1234 + rank)
    x = torch.randn(batch // world_size, seq, dim, generator=g, device=device)
    y = torch.randn(batch // world_size, seq, dim, generator=g, device=device)
    return x, y

def bench(ddp_cls, model, input_ids, targets, steps=10):
    ddp_model = ddp_cls(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(2):
        optimizer.zero_grad()
        loss = ddp_model(input_ids).sum()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ddp_model(input_ids).sum()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / steps

def main(rank, world_size):
    setup(rank, world_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)   # 保证各 rank 初始参数相同（虽然 DDP 会广播）

    x, y = make_data(rank, world_size, device=device)

    torch.manual_seed(42)
    base = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 128),
    ).to(device)
    init_state = {k: v.clone() for k, v in base.state_dict().items()}

    def make_model():
        m = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 128),
        ).to(device)
        m.load_state_dict(init_state)
        return m

    results = {}
    for name, cls in [("Naive", NaiveDDP), ("Flat", FlatDDP), ("Overlap", OverlapDDP)]:
        model = make_model()
        ms = bench(cls, model, x, y)
        results[name] = ms
        if rank == 0:
            print(f"[{name:>8}] {ms:.2f} ms/step", flush=True)

    if rank == 0:
        print("\n=== Summary (rank 0) ===")
        for name, ms in results.items():
            print(f"{name:>8}: {ms:.2f} ms/step")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)