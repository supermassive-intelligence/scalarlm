from tokenformer.llama_tokenformer_decoder_layer import LlamaTokenformerDecoderLayer
import logging

logger = logging.getLogger(__name__)

def replace_decoder_layers(model, custom_layer_class):
    # Replace decoder layers with custom layers
    for i, layer in enumerate(model.model.layers):
        new_layer = custom_layer_class(model.config, i)
        new_layer.load_state_dict(layer.state_dict(), strict=False)
        model.model.layers[i] = new_layer
    return model
        

def create_llama_tokenformer_model(model):
    model = replace_decoder_layers(model, LlamaTokenformerDecoderLayer)
    # Set requires_grad to False for all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Set requires_grad to True for tokenformer params and lm_head
    for name, param in model.named_parameters():
        if "tokenformer" in name or "lm_head" in name:
            param.requires_grad = True
        logger.debug(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    
    return model

