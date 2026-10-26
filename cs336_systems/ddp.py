import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

def setup(rank, world_size):
    print(f"Rank {rank} entering setup", flush=True)
    os.environ["MASTER_ADDR"]  = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # try:
    #     dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #     print(f"[Rank {rank}] init done", flush=True) # 新增
    # except Exception as e:
    #     print(f"[Rank {rank}] init failed: {e}", flush=True) # 新增
    #     raise

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank}: {data}")

class NaiveDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        if dist.is_initialized():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        if not dist.is_initialized():
            return None
        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= world_size

class OverlapDDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.handles = []
        if dist.is_initialized():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

            for p in self.module.parameters():
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(self._hook)
    
    def _hook(self, p):
        if dist.is_initialized() and p.grad is not None:
            handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, p))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        if not dist.is_initialized():
            return None
        world_size = dist.get_world_size()
        for handle, p in self.handles:
            handle.wait()
            p.grad.data /= world_size
        self.handles.clear()

class FlatDDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        if dist.is_initialized():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        if not dist.is_initialized():
            return None
        world_size = dist.get_world_size()

        grads = [p.grad for p in self.module.parameters() if p.grad is not None]
        if not grads:
            return None
        flat = _flatten_dense_tensors(grads)

        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= world_size

        unflat = _unflatten_dense_tensors(flat, grads)
        for g, new_g in zip(grads, unflat):
            g.copy_(new_g)


if __name__ == "__main__":
    world_size = 4
    mp.spawn(distributed_demo, args=(world_size, ), nprocs=world_size, join=True)
