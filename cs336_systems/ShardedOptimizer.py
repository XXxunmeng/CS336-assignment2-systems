import torch
import torch.distributed as dist
from typing import Type

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.local_params = []
        self.local_optimizer = None
        super().__init__(params, defaults=kwargs)

    def add_param_group(self, param_group):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        super().add_param_group(param_group)

        for i, p in enumerate(param_group['params']):
            if i % world_size == rank:
                self.local_params.append(p)

        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls(self.local_params, **self.kwargs)
        else:
            self.local_optimizer.add_param_group(
                {"params": [p for i, p in enumerate(param_group['params']) if i % world_size == rank],
                **{k: v for k, v in param_group.items() if k != "params"}}
            )

    def step(self, closure=None):
        if self.local_optimizer is None:
            return None
            
        loss = self.local_optimizer.step(closure)
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    src = i % world_size
                    dist.broadcast(p.data, src=src)
        return loss