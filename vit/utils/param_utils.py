import math 

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import torch.distributed as dist

import re

from helper_functions.distributed import print_at_master


# From the long param name like "module.1.attention.query.weight" extract layer number(i.e. 1)
# and component name (i.e. query)
def get_layer_number_and_family(param_name):
    layer_match = re.search(r'\.layer\.(\d+)\.', param_name)
    family_match = re.search(r'\.(query|key|value|output|dense|proj|fc)(\.|$)', param_name)
    
    layer = int(layer_match.group(1)) if layer_match else None
    family = family_match.group(1) if family_match else None
    
    return layer, family

# Calcualte the l2 distance between the tensors of a given param at two stages.
# Convert the l2 distance to the percentage change in the value.
def get_percent_change_in_norm(param_name, v1,v2, eps=1e-8):
    layer_number, layer_family = get_layer_number_and_family(param_name)
    percent_change =abs((v2-v1)/v1) * 100

    return layer_number, layer_family, percent_change


# Given collection of all the params e.g. [module.1.query.weight, module.1.key.weight, module.1.fc.weight ,...]
# Seperate the family of these params and cluster it . e.g. {'query':[module.1.query.weight], 'key': [module.1.key.weight], 'fc': [module.1.fc.weight]}
def cluster_param_based_on_family(all_param_names):
    param_family_clustering = {}

    for param_name in all_param_names:
        _, param_family = get_layer_number_and_family(param_name)

        if param_family in param_family_clustering:
            param_family_clustering[param_family].append(param_name)
        else:
            param_family_clustering[param_family] = [param_name]
    
    return param_family_clustering


# Build a dict that stores the metric changes of params belonging to each family of param.
# e.g. {'query':{'module.1.query.weight' : 10, 'module.7.query.weight': 5}, 'key': {....}}
def get_family_param_metric_change(all_params, family_groups):
    family_params_metric_change = {}

    for param_family in family_groups:
        for param_name in family_groups[param_family]:
            _, family, pc = get_percent_change_in_norm(param_name, all_params[-2][param_name], all_params[-1][param_name])

            if family in family_params_metric_change:
                family_params_metric_change[family][param_name] = pc
            else:
                family_params_metric_change[family] = {}
                family_params_metric_change[family][param_name] = pc
    
    return family_params_metric_change

# Normalize the weight changes seperately for the param based on the family it belongs to.
# For example if a param belongs to 'query' family, obtain min and max from only the params in that family for normalization.
def get_family_wise_normalized_metrics(family_params_metric_change):
    param_dict = {}

    for family in family_params_metric_change:
        all_metric_changes = torch.tensor(list(family_params_metric_change[family].values()))

        family_params_metrics = family_params_metric_change[family]

        for param in family_params_metrics:
            param_dict[param] = (family_params_metrics[param] - all_metric_changes.min())/(all_metric_changes.max() - all_metric_changes.min())
    
    return param_dict

# Consider all the param name irrespective of the family they belong to
# Get max and min across all the params and perform normalization.
def get_overall_normalized_metrics(all_params):
    all_param_names = all_params[0].keys()
    param_dict = {}

    for param_name in all_param_names:
        _, family, pc = get_percent_change_in_norm(param_name, all_params[-2][param_name], all_params[-1][param_name])

        param_dict[param_name] = pc
    
    max_metric_val = max(list(param_dict.values()))
    min_metric_val = min(list(param_dict.values()))

    for param_name in param_dict:
        param_dict[param_name] = (param_dict[param_name] - min_metric_val)/(max_metric_val - min_metric_val)

    return param_dict

def is_one_of_module(module_name, comparisions):
    results = []

    for comparision in comparisions:
        result = re.search(re.escape(f"{comparision}") + r"$", module_name)

        if result:
            results.append(True)
            continue

        results.append(False)

    return any(results)

def get_selected_modules_norms(model, selected_modules):
    selected_modules_weights_norms = {}
    selected_modules_grad_norms = {}

    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and is_one_of_module(name, selected_modules):
            if hasattr(module, 'weight') and module.weight.grad is not None:
                selected_modules_weights_norms[name] = module.weight.data.norm().item()                
                selected_modules_grad_norms[name] = module.weight.grad.norm().item()
    
    return selected_modules_weights_norms, selected_modules_grad_norms

def get_norms_from_selected_lora_modules(model, selected_module_names):
    weight_norms = {}
    grad_norms = {}

    for name, module in model.module.named_modules():
        if name in selected_module_names:
            # Base layer
            if hasattr(module, "base_layer"):
                base = module.base_layer
                if hasattr(base, "weight") and base.weight.grad is not None:
                    key = f"{name}.base_layer"
                    weight_norms[key] = base.weight.data.norm().item()
                    grad_norms[key] = base.weight.grad.norm().item()

            # LoRA A
            if hasattr(module, "lora_A") and module.lora_A['default'].weight.grad is not None:
                key = f"{name}.lora_A"
                weight_norms[key] = module.lora_A['default'].weight.data.norm().item()
                grad_norms[key] = module.lora_A['default'].weight.grad.norm().item()

            # LoRA B
            if hasattr(module, "lora_B") and module.lora_B['default'].weight.grad is not None:
                key = f"{name}.lora_B"
                weight_norms[key] = module.lora_B.weight['default'].data.norm().item()
                grad_norms[key] = module.lora_B.weight['default'].grad.norm().item()

    return weight_norms, grad_norms

def average_module_norms_across_gpus(norm_dict, local_rank):
    averaged = {}
    for name, val in norm_dict.items():
        val = torch.tensor(val).to(torch.device(f"cuda:{local_rank}"))
        val_clone = val.clone()
        dist.all_reduce(val_clone, op=dist.ReduceOp.SUM)
        val_clone /= dist.get_world_size()
        averaged[name] = val_clone
    return averaged