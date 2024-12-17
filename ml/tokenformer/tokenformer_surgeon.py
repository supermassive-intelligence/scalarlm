from abc import abstractmethod, ABC
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class TokenformerMLPAdapter(nn.Module):
    def __init__(self, layer, hidden_size):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
    
        self.tokenformer_k = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.tokenformer_v = nn.Parameter(torch.zeros(hidden_size, hidden_size))

    # Call layer with all inputs and kwargs
    def forward(
        self,
        query: torch.Tensor
    ):
        base_layer_results = self.layer(query)
        
        tokenformer_results = torch.nn.functional.scaled_dot_product_attention(
            query=query, key=self.tokenformer_k, value=self.tokenformer_v,
            attn_mask=None,
            is_causal=False # should be false for tokenformer
        )
        
        # sum the two outputs
        layer_and_adaptor_sum = base_layer_results + tokenformer_results
        return layer_and_adaptor_sum    

class TokenformerAttentionAdapter(nn.Module):
    def __init__(self, layer, hidden_size):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
    
        self.tokenformer_k = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.tokenformer_v = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))

    def forward(
        self,
        query,
        base_layer_results
    ) -> torch.Tensor:

        tokenformer_results = torch.nn.functional.scaled_dot_product_attention(
            query=query, 
            key=self.tokenformer_k, 
            value=self.tokenformer_v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False # should be false for tokenformer
        )
        
        # sum the two outputs
        layer_and_adaptor_sum = base_layer_results + tokenformer_results
        return layer_and_adaptor_sum   
    
    
class TokenformerSurgeon(ABC):
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def _is_attn_layer(self, layer_name):
        return layer_name.split('.')[-1] == "attn"

    def _is_mlp_layer(self, layer_name):
        return "mlp" in layer_name.split('.')[-1]

    def _recursive_setattr(self, obj, attr, value):
        attr = attr.split(".", 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            self._recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    def _try_to_update_mlp(self, name, layer):
        """Try to wrap the layer with a TokenformerMLPAdaptor."""
        if not self._is_mlp_layer(name):
            return

        logger.info(f"Wrapping layer {name} with TokenformerMLPAdaptor")

        # Wrap the layer with a TokenformerMLPAdapter
        self._recursive_setattr(self.model, name, TokenformerMLPAdapter(layer, self.model.config.hidden_size))

    @abstractmethod
    def _try_to_update_attn(self, name, layer):
        pass
    
    def insert_adapter_modules(self):
        # Add tokenformer adapters for mlp and attention
        for name, layer in self.model.named_modules():
            self._try_to_update_mlp(name, layer)
            self._try_to_update_attn(name, layer)
        
        return self.model

        