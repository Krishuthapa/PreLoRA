import torch

from functools import partial

from helper_functions.distributed import print_at_master

@torch.no_grad()
def random_pruning_(tensor, prune_ratio):
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor.mul_(random_pruning_mask)


@torch.no_grad()
def magnitude_pruning_(tensor, prune_ratio):
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor.mul_(mask.to(dtype=tensor.dtype))

def optimizer_reset(
    optimizer,
    reset_params,
    optimizer_state_keys,
    optimizer_magnitude_pruning=0.6,
):

    pruning_fn = partial(magnitude_pruning_, prune_ratio=optimizer_magnitude_pruning)
    
    n_zeros = 0
    n_total = 0

    optimizer_state = optimizer.state
    for p in reset_params:
        param_state = optimizer_state[p]
        if len(param_state) == 0:
            continue
        
        for key in optimizer_state_keys:
            if key in param_state:
                pruning_fn(param_state[key])
                n_total += param_state[key].numel()
                n_zeros += torch.sum(param_state[key] == 0).item()

    _zeroed = n_zeros / (1e-7 + n_total) * 100
    print_at_master(f"Pruned {_zeroed:.2f}% of optimizer state.")
