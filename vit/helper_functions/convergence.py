import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from utils.param_utils import get_percent_change_in_norm, cluster_param_based_on_family, get_family_param_weight_change
from utils.weight_decay_utils import powerlaw_tail_custom, exponential_decay, linear_decay

from helper_functions.distributed import print_at_master


# Use power law to associalte larger weight to last layers when
# computing the overall percentage change for each modules like
# ('query','key', etc) for all the layers.
def get_overall_percentage_change(component_change, decay='none', last_weight = 0.5):
    n_layers = max(component_change.keys()) + 1

    if decay == 'power':
        weights_decay = powerlaw_tail_custom(n_layers, last_weight)
    elif decay == 'exp':
        weights_decay = exponential_decay(n_layers, last_weight)
    elif decay == 'lin':
        weights_decay = linear_decay(n_layers, last_weight)
    else:
        weights_decay = np.array([ 1 for _ in range(n_layers)]) / (n_layers)
    
    total_percent_change =0

    print("Weights decay")
    print_at_master(weights_decay)

    for key in sorted(component_change.keys()):
        total_percent_change += weights_decay[key] * component_change[key]

    return total_percent_change


# Checks if loss values have converged over consecutive checkpoints by ensuring
# percentage changes between losses are within a specified threshold.
def check_loss_convergence(losses, threshold, checking_step):
    keys = sorted([int(key) for key in losses.keys()])
    are_consecutive_keys = all(b-a == checking_step for a,b in zip(keys,keys[1:]))

    assert are_consecutive_keys, f"Losses are not stored for {len(losses)} consecutive windows of {checking_step} iterations each."
    
    percent_changes = [(abs(losses[b]-losses[a])/float(losses[a]) * 100) for a,b in zip(keys,keys[1:])]

    print_at_master("Percent changes")
    print_at_master(percent_changes)

    return all([change <= threshold for change in percent_changes])


# Params states have data in form {'module.1.query.weight' : tensor(dxd), 'module.1.key.weight' : tensor(dxd) ... }
def check_weight_convergence(weights, threshold):
    weights_list = [weights[key] for key in sorted(weights.keys())]
    all_modules = weights_list[0].keys()

    convergence_result_holder = []
    
    # We check the convergence after l (defualt:3)  consecutive windows.
    for index in range(len(weights_list)-1):
        module_layer_wc = {}

        for module_name in all_modules:
            layer_num, module_family, wc = get_percent_change_in_norm(module_name, weights_list[index][module_name], weights_list[index+1][module_name])
            
            if module_family in module_layer_wc:
                module_layer_wc[module_family][layer_num] = wc
            else:
                module_layer_wc[module_family] = {}
                module_layer_wc[module_family][layer_num] = wc
        
        # Gets overall change in the params for each module, the overall change is summation based on the power law
        # importance associated with each layer.
        # {'query': 15 , 'key': 12}

        for module_family in module_layer_wc.keys():
            module_change = get_overall_percentage_change(module_layer_wc[module_family], decay='power', last_weight= 0.45)
            
            convergence_result_holder.append(module_change < threshold)

    return all(convergence_result_holder)

def check_and_remove_unnecessary_metric(metric_store, balance = 3):
    all_available_timesteps = sorted(metric_store.keys())

    if len(all_available_timesteps) <= balance:
        return metric_store

    new_metric_store = {}

    for timestep in range(len(all_available_timesteps)- balance, len(all_available_timesteps)):
        new_metric_store[all_available_timesteps[timestep]] = metric_store[all_available_timesteps[timestep]]
    
    return new_metric_store


def check_partial_convergence(losses, weights, thr_loss = 10, thr_weight = 5, checking_step = 1000, checking_window = 3):
    is_loss_converged = False
    is_weight_converged = False

    if len(losses.keys()) == checking_window and len(weights.keys()) == checking_window:
        is_loss_converged = check_loss_convergence(losses, thr_loss, checking_step)
        is_weight_converged = check_weight_convergence(weights, thr_weight)

    return is_loss_converged and is_weight_converged

# Code for the hook injected to check the activation of base weights and lora weights.
def check_activation_change(module_name, threshold, hook_ctrl):
    def hook(module, input, output):
        is_base_frozen = any(not param.requires_grad for param in module.base_layer.parameters())
        
        if is_base_frozen:
            return 
        
        if not hook_ctrl.should_run():
            return
        
        x = input[0]
        base_out = module.base_layer(x)

        if isinstance(module.lora_A, torch.nn.ModuleDict):
            A = module.lora_A["default"]
            B = module.lora_B["default"]
        else:
            A = module.lora_A
            B = module.lora_B

        if isinstance(module.lora_magnitude_vector, torch.nn.ModuleDict):
            scaling = module.lora_magnitude_vector['default'].weight
        else:
            scaling = int(module.scaling['default'])

        lora_out = B(A(x)) * scaling
    
        lora_act_norm = torch.linalg.vector_norm(lora_out, p=2).item()
        base_act_norm = torch.linalg.vector_norm(base_out, p=2).item()

        ratio = (lora_act_norm / base_act_norm) * 100 if base_act_norm > 0 else 0.0

        if ratio < threshold:
            print(f"Freezing back {module_name} and freezing base_layer.")

            for param in module.base_layer.parameters():
                param.requires_grad = False

    return hook