import torch
import torch.distributed as dist
import torch.nn as nn
from cs336_basics.transformer import Embedding, Linear, RMSNorm

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, world_size, full_shape, numel, padded_numel):
        ctx.world_size = world_size
        ctx.shard_numel = shard.numel()
        ctx.shard_dtype = shard.dtype
        ctx.numel = numel
        ctx.padded_numel = padded_numel
        if world_size == 1 or not dist.is_initialized():
            return shard[:numel].reshape(full_shape)
        gathered = [torch.empty_like(shard) for _ in range(world_size)]
        dist.all_gather(gathered, shard)
        return torch.cat(gathered)[:numel].reshape(full_shape)

    @staticmethod
    def backward(ctx, grad_full):
        if ctx.world_size == 1 or not dist.is_initialized():
            grad_shard = grad_full.flatten()[:ctx.shard_numel]
            return grad_shard.to(ctx.shard_dtype), None, None, None, None

        grad_flat = grad_full.flatten()
        if grad_flat.numel() < ctx.padded_numel:
            grad_flat = torch.cat([grad_flat, torch.zeros(ctx.padded_numel - grad_flat.numel(), device=grad_flat.device, dtype=grad_flat.dtype)])

        scattered = torch.empty(ctx.shard_numel, device=grad_flat.device, dtype=grad_flat.dtype)
        grad_2d = grad_flat.reshape(ctx.world_size, ctx.shard_numel)
        dist.reduce_scatter_tensor(scattered, grad_flat, op=dist.ReduceOp.SUM)
        return (scattered / ctx.world_size).to(ctx.shard_dtype), None, None, None, None

class FSDP(torch.nn.Module):
    def __init__(self, module, compute_dtype=None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.sharded_info = {}

        self._shard_all_params()

        if dist.is_initialized():
            self._register_forward_hooks() 

        # self._register_backward_hooks()

    def _shard_all_params(self):
        seen = {}
        for name, module in self.module.named_modules():
            if isinstance(module, (Embedding, Linear)):

                param = module.weight
                full_name = f"{name}.weight" if name else "weight"
                if id(param) in seen:
                    new_param, info = seen[id(param)]
                    module.weight = new_param
                    self.sharded_info[full_name] = info
                    continue

                original_shape = param.shape
                flat = param.data.flatten()
                numel = flat.numel()
                shard_size = (numel + self.world_size - 1) // self.world_size
                padded_numel = shard_size * self.world_size
                if padded_numel > numel:
                    flat = torch.cat([flat, torch.zeros(padded_numel - numel, device=flat.device, dtype=flat.dtype)])
                start = self.rank * shard_size
                shard = flat[start:start + shard_size].clone()
    
                info = {
                    "original_shape": original_shape,
                    "shard_size": shard_size,
                    "padded_numel": padded_numel,
                    "numel": numel
                }

                self.sharded_info[full_name] = info
                new_param = nn.Parameter(shard, requires_grad=param.requires_grad)
                module.weight = new_param
                seen[id(param)] = (new_param, info)
        print(f"[rank {self.rank}] sharded_info: {sorted(self.sharded_info.keys())}", flush=True)

    def _register_forward_hooks(self):
        for name, module in self.module.named_modules():
            if isinstance(module, (Embedding, Linear)):
                full_name = f"{name}.weight" if name else "weight"
                module.register_forward_pre_hook(self._make_gather_hook(full_name))
                module.register_forward_hook(self._make_release_hook())
                    

    def _make_gather_hook(self, full_name):
        def hook(module, *args):
            # full_name = f"{layer_name}.{name}"
            if full_name not in self.sharded_info:
                return
            info = self.sharded_info[full_name]
            shard = module.weight
            full = AllGather.apply(shard, self.world_size, info['original_shape'], info['numel'], info['padded_numel'])

            if self.compute_dtype is not None:
                full = full.to(self.compute_dtype)
            module._orig_weight = module.weight
            object.__setattr__(module, "weight", full)
            # module.weight = full
        return hook

    def _make_release_hook(self):
        def hook(module, input, output):
            if hasattr(module, "_orig_weight"):
                # module.weight = module._orig_weight
                object.__setattr__(module, "weight", module._orig_weight)
                del module._orig_weight
        return hook

    def _register_backward_hooks(self):
        for name, module in self.module.named_modules():
            if isinstance(module, (Embedding, Linear)):
                full_name = f"{name}.weight" if name else "weight"
                if full_name in self.sharded_info:
                    module.register_full_backward_pre_hook(self._make_gather_hook(full_name))
                    # module.register_full_backward_hook(self._make_release_hook())
                    # module.weight.register_post_accumulate_grad_hook(self._make_reduce_hook(full_name))

    def _make_reduce_hook(self, full_name):
        def hook(p):
            info = self.sharded_info[full_name]
            grad = p.grad.data.flatten()

            if grad.numel() < info["padded_numel"]:
                grad = torch.cat([grad, torch.zeros(info["padded_numel"] - grad.numel(), device=grad.device, dtype=grad.dtype)])

            scattered = torch.empty(info['shard_size'], device=grad.device, dtype=grad.dtype)
            dist.reduce_scatter_tensor(scattered, grad.reshape(self.world_size, -1), op=dist.ReduceOp.SUM)
            p.grad.data = scattered / self.world_size

        return hook
            

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if not dist.is_initialized():
            return

        seen_ids = set()
        print(f"[rank {dist.get_rank()}] finish_gradient_synchronization called", flush=True)
        for name, param in self.module.named_parameters():
            if name in self.sharded_info or param.grad is None:
                continue
            if id(param) in seen_ids:
                continue

            seen_ids.add(id(param))
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.world_size

    def gather_full_params(self):
        modules = dict(self.module.named_modules())
        result = {}
        for name, info in self.sharded_info.items():
            module_name, _ = name.rsplit('.', 1)
            module = modules[module_name]
            shard = module.weight.data
            gathered = [torch.empty_like(shard) for _ in range(self.world_size)]
            dist.all_gather(gathered, shard)
            full_gathered = torch.cat(gathered)[:info["numel"]].reshape(info['original_shape'])
            result[name] = full_gathered
        for name, param in self.module.named_parameters():
            if name not in result:
                result[name] = param.data.clone()
        return result
    