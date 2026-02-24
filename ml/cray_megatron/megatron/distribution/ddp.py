from gpu_aware_mpi import allreduce, get_size

import torch.nn as nn


class DDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def backward_sync(self):
        for param_name, param in self.model.named_parameters(recurse=False):
            if param.requires_grad and param.grad is not None:
                allreduce(param.grad)
