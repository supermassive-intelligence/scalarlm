import torch
import torch.nn as nn
import numpy as np
from mpi4py import MPI

import time

import logging

logger = logging.getLogger(__name__)

class FSDPLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.comm = MPI.COMM_WORLD
        self.world_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.backward_pre_handle = self.module.register_full_backward_pre_hook(
            self.full_backward_pre_hook
        )
        self.backward_handle = self.register_full_backward_hook(
            self.synchronize_gradients_hook
        )
        self.forward_pre_handle = self.module.register_forward_pre_hook(
            self.forward_pre_hook
        )
        self.forward_handle = self.module.register_forward_hook(self.forward_hook)

        self.original_weight_shape = None
        self.original_bias_shape = None
        self.shard_parameters(module)

    def forward_hook(self, module, input, output):
        self.shard_parameters(module)
        return output
   
    def all_gather(self, tensor):
        """
        Gathers tensors from all ranks into a single concatenated tensor,
        using Allgather and correctly removing padding.
        """
        if tensor is None:
            return None

        tensor_numpy = tensor.detach().to(torch.float32).cpu().numpy()
        original_shape = tensor_numpy.shape
        original_size = tensor_numpy.size

        # Determine the maximum size of tensor across all ranks, for padding purposes
        max_size = self.comm.allreduce(tensor_numpy.size, op=MPI.MAX)

        # Calculate padding size, if needed.
        padding_size = max_size - tensor_numpy.size
        if padding_size > 0:
            pad_shape = list(original_shape)
            pad_shape[-1] = padding_size
            padding = np.zeros(pad_shape, dtype=tensor_numpy.dtype)
            tensor_numpy = np.concatenate((tensor_numpy, padding.flatten())).astype(tensor_numpy.dtype)
            tensor_numpy = tensor_numpy.reshape(-1)  # Flatten again, required for allgather

        # Allocate the correctly sized buffer for Allgather
        gathered_numpy = np.empty(max_size * self.world_size, dtype=tensor_numpy.dtype)
        
        start = time.time()
        self.comm.Allgather(tensor_numpy, gathered_numpy)
        end = time.time()

        #logger.info(f"Rank {self.rank} all_gather time: {end - start}, bandwidth: {tensor_numpy.nbytes / (end - start) / 1e9} GB/s on tensor {tensor.shape}")

        # Now we need to extract the original tensors, accounting for potential padding.
        all_original_sizes = self.comm.allgather(original_size)  
        all_tensors = []
        offset = 0  
        for rank in range(self.world_size):
            original_size_rank = all_original_sizes[rank]
            tensor_rank = gathered_numpy[offset:offset + original_size_rank]  
            all_tensors.append(tensor_rank)
            offset += max_size  

        # Concatenate all the tensors
        concatenated_numpy = np.concatenate(all_tensors)
        return torch.from_numpy(concatenated_numpy).to(tensor.device).to(tensor.dtype)
    
    def forward_pre_hook(self, module, input):
        self.gather_all_parameters(module)
        return input

    def gather_all_parameters(self, module):
        if hasattr(module, "weight"):
            full_weight = self.all_gather(module.weight)
            module.weight.data = full_weight.view(self.original_weight_shape)
        if hasattr(module, "bias") and module.bias is not None:
            full_bias = self.all_gather(module.bias)
            module.bias.data = full_bias.view(self.original_bias_shape)
            
    def reduce_scatter(self, param):
        if param.grad is not None:
            grad = param.grad.data
            world_size = self.world_size
            rank = self.rank
            
            # Ensure contiguous tensor and convert to FP32
            grad_contig = grad.contiguous().to(torch.float32)
            grad_numpy = grad_contig.detach().cpu().numpy().flatten()
            original_dtype = grad.dtype
            
            # Calculate padding for equal division
            total_elements = grad_numpy.size
            padded_size = ((total_elements + world_size - 1) // world_size) * world_size
            local_size = padded_size // world_size
            
            # Pad array if necessary
            if padded_size != total_elements:
                padded_grad = np.zeros(padded_size, dtype=grad_numpy.dtype)
                padded_grad[:total_elements] = grad_numpy
            else:
                padded_grad = grad_numpy

            # Prepare receive buffer
            local_grad = np.zeros(local_size, dtype=grad_numpy.dtype)
            
            # MPI collective operation
            self.comm.Reduce_scatter(padded_grad, local_grad, op=MPI.SUM)
            
            # Trim padding from last rank
            if rank == world_size - 1 and padded_size != total_elements:
                valid_elements = total_elements - (world_size - 1) * local_size
                local_grad = local_grad[:valid_elements]
            
            # logger.info(f"Rank {rank}: Reduced {grad_numpy.shape}->{local_grad.shape}")        
            param.grad.data = torch.from_numpy(local_grad).to(device=grad.device).to(original_dtype)


    def full_backward_pre_hook(self, module, grad_output):
        self.gather_all_parameters(module)
        return grad_output


    def synchronize_gradients_hook(self, module, grad_input, grad_output):
        
        if hasattr(module, "weight"):    
            self.reduce_scatter(module.weight)
        if hasattr(module, "bias") and module.bias is not None:  
            self.reduce_scatter(module.bias)
        
        return grad_input

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def remove_hooks(self):
        if hasattr(self, "backward_handle"):
            self.backward_handle.remove()
        if hasattr(self, "backward_pre_handle"):
            self.backward_pre_handle.remove()
        if hasattr(self, "forward_pre_handle"):
            self.forward_pre_handle.remove()

    def shard_parameters(self, module):
        if hasattr(module, "weight"):
            self.original_weight_shape = module.weight.data.shape
            module.weight.data = self.shard_tensor(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            self.original_bias_shape = module.bias.data.shape
            module.bias.data = self.shard_tensor(module.bias)

    def shard_tensor(self, tensor):
        flat_view = tensor.view(-1)
        total_elements = flat_view.numel()
        padded_size = ((total_elements + self.world_size - 1) // self.world_size) * self.world_size
        local_size = padded_size // self.world_size
        start = self.rank * local_size
        end = start + local_size
        
        # Handle the last rank correctly when padding is needed
        if end > total_elements:
           end = total_elements

        shard = flat_view[start:end]

        return shard

class SimpleFSDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._wrap_layers(model)

    def _wrap_layers(self, module):
        for name, child in module.named_children():
            if hasattr(child, "tokenformer_k") or hasattr(child, "tokenformer_v"):
                wrapped = FSDPLayer(child)
                setattr(module, name, wrapped)
                self._wrap_layers(child)
            elif list(child.children()):
                self._wrap_layers(child)
            else:
                wrapped = FSDPLayer(child)
                setattr(module, name, wrapped)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def remove_all_hooks(self):
        for module in self.model.modules():
            if isinstance(module, FSDPLayer):
                module.remove_hooks()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def unwrap_model(self):
        config = self.model.config
        unwrapped_model = type(self.model)(config)
        unwrapped_state_dict = {}

        for name, module in self.model.named_modules():
            if isinstance(module, FSDPLayer):
                if hasattr(module.module, "weight"):
                    full_weight = module.all_gather(module.module.weight)
                    unwrapped_state_dict[f"{name}.weight"] = full_weight
                if hasattr(module.module, "bias") and module.module.bias is not None:
                    full_bias = module.all_gather(module.module.bias)
                    unwrapped_state_dict[f"{name}.bias"] = full_bias

        # Load the gathered state dict into the new model
        unwrapped_model.load_state_dict(unwrapped_state_dict, strict=False)

        # Fix for pad_token_id
        unwrapped_model.config.pad_token_id = unwrapped_model.config.eos_token_id
        if hasattr(unwrapped_model, "generation_config"):
            unwrapped_model.generation_config.pad_token_id = (
                unwrapped_model.config.eos_token_id
            )

        return unwrapped_model