import torch
import subprocess as sp

def get_free_gpu() -> torch.device:

    # check if cuda is available
    if torch.cuda.is_available():
        # run nvidia-smi command to get data on free memory
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = (
            sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
        )
        # extract memory values
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        # return gpu with the most free memory
        gpu = f"cuda:{memory_free_values.index(max(memory_free_values))}"

    # if cuda isn't available, run on cpu
    else:
        gpu = "cpu"

    # return the device with the most free memory
    return torch.device(gpu)