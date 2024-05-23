import torch

def cuda_memory():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))