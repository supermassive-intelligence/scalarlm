from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# MPI_Allgather
send_data = np.array([rank + 1], dtype='i')  
recv_data = np.empty(size, dtype='i')          

comm.Allgather(send_data, recv_data)

if rank == 0:
    print(f"Rank {rank} received data after Allgather: {recv_data}")

# MPI_Reduce_scatter - each rank sends size integers, receives 1 integer
send_data = np.array([(rank + 1) * i for i in range(1, size + 1)], dtype='i')  
recv_counts = np.ones(size, dtype='i')                                     
recv_data = np.empty(1, dtype='i')                   

comm.Reduce_scatter(send_data, recv_data, recvcounts=recv_counts, op=MPI.SUM)

print(f"Rank {rank} received data after Reduce_scatter: {recv_data}")
