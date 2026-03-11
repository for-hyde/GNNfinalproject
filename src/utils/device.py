import torch
import subprocess as sp
from collections import OrderedDict


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


def load_model(model, state_dict_path):
    state_dict = torch.load(state_dict_path)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model
