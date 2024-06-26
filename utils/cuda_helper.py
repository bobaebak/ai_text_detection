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

def print_total_gpu_memory():
    # Get the device properties for the current GPU
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(device_props.name)

    # Print the total GPU memory in gigabytes
    total_gpu_memory = device_props.total_memory / (1024**3)
    print(f"Total GPU memory: {total_gpu_memory:.2f} GB")

def clear_gpu():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def cuda_memory():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))