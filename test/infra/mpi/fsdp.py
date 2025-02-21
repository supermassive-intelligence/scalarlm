from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import math
import random
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a fixed seed
seed_all(42)

class FSDPLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.rank = rank
        self.world_size = world_size

    def all_gather(self, tensor):
        if tensor is None:
            return None
        tensor_numpy = tensor.detach().numpy()
        gathered = np.zeros([self.world_size] + list(tensor_numpy.shape), dtype=tensor_numpy.dtype)
        comm.Allgather(tensor_numpy, gathered)
        return torch.from_numpy(gathered).to(tensor.device)

    def forward(self, x):
        if hasattr(self.layer, 'weight'):
            full_weight = self.all_gather(self.layer.weight)
            self.layer.weight.data = full_weight[self.rank]
        if hasattr(self.layer, 'bias'):
            full_bias = self.all_gather(self.layer.bias)
            self.layer.bias.data = full_bias[self.rank]
        return self.layer(x)

    def synchronize_gradients(self):
        def reduce_scatter(param):
            if param.grad is not None:
                grad = param.grad.data
                local_size = grad.numel() // self.world_size
                start = self.rank * local_size
                end = start + local_size if self.rank < self.world_size - 1 else grad.numel()
                
                local_grad = torch.zeros(end - start, dtype=grad.dtype, device=grad.device)
                
                comm.Reduce_scatter(grad.view(-1).numpy(), local_grad.numpy(), op=MPI.SUM)
                
                grad.view(-1)[start:end] = local_grad
                grad /= self.world_size
                param.grad.data = grad

        if hasattr(self.layer, 'weight'):
            reduce_scatter(self.layer.weight)
        if hasattr(self.layer, 'bias'):
            reduce_scatter(self.layer.bias)

class FSDPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(self._get_fsdp_layers(model))

    def _get_fsdp_layers(self, model):
        return [FSDPLayer(module) for module in model.modules() if self._is_leaf_module(module)]

    @staticmethod
    def _is_leaf_module(module):
        return not list(module.children())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, loss):
        loss.backward()
        for layer in reversed(self.layers):
            layer.synchronize_gradients()


# Define a simple transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super(SimpleTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) 
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_decoder(src, src, tgt_mask=None)
        output = self.decoder(output)
        return output  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def test_sequential_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)
    
    fsdp_model = FSDPModel(model)
    
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.01)
    for epoch in range(10):
        input_data = torch.randn(32, 10, device=device)
        output = fsdp_model(input_data)
        
        target = torch.randn(32, 5, device=device)
        loss = nn.functional.mse_loss(output, target)
        
        optimizer.zero_grad()
        fsdp_model.backward(loss)
        optimizer.step()
        
        if rank == 0:
            logger.debug(f"Epoch {epoch}, Loss: {loss.item()}")

def test_transformer_model():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model parameters 
    ntokens = 1000  # size of vocabulary
    emsize = 64  # embedding dimension
    d_hid = 64  # dimension of the feedforward network model
    nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability

    # Create model and move to device
    model = SimpleTransformer(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    # Ensure identical initialization across ranks
    for param in model.parameters():
        comm.Bcast(param.data.numpy(), root=0)

    # Wrap model with FSDP
    model = FSDPModel(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create synthetic dataset
    def create_dataset(num_samples, seq_length):
        # Use a fixed seed for dataset creation
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        data = torch.randint(0, ntokens, (num_samples, seq_length), dtype=torch.long)
        target = torch.randint(0, ntokens, (num_samples, seq_length), dtype=torch.long)
        torch.set_rng_state(rng_state)
        return TensorDataset(data, target)

    # Create DataLoader with DistributedSampler for MPI ranks
    num_samples = 1000  
    seq_length = 20  
    batch_size = 16  
    dataset = create_dataset(num_samples, seq_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    # Training loop
    def train(model, data_loader, optimizer, criterion, epochs, device):
        model.train()
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            total_loss = 0

            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)  
                loss = criterion(output.view(-1, ntokens), target.view(-1))  

                model.backward(loss)
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0 and rank == 0:
                    logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Synchronize total_loss across all ranks
            total_loss = comm.allreduce(total_loss, op=MPI.SUM)
            if rank == 0:
                logger.debug(f"Epoch {epoch+1} completed. Average Loss: {total_loss / (len(data_loader) * world_size):.4f}")

    # Run training
    train(model, train_loader, optimizer, criterion, epochs=5, device=device)

    # Finalize MPI environment
    comm.Barrier()
    if rank == 0:
        logger.debug("Training complete.")

if __name__ == "__main__":
    test_sequential_model()

# mpirun --allow-run-as-root -np 2 --oversubscribe python fsdp.py
