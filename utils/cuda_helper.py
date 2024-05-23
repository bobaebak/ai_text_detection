import torch


def print_gpu_usage():
    # Check current GPU memory usage
    current_memory = torch.cuda.memory_allocated() / (1024**2)  # Convert to MB
    print(f"Current GPU memory usage: {current_memory:.2f} MB")

    # Check maximum GPU memory usage
    max_memory = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Maximum GPU memory usage: {max_memory:.2f} MB")

def print_gpu_devie_name():
    num_gpus = torch.cuda.device_count()

    # Print the GPU names
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"cuda {i}: {gpu_name}")

def clear_gpu():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
