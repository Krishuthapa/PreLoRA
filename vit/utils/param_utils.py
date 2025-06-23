import math 

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

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
    numerators = torch.norm(v2-v1, p=2).detach().clone()
    denominators = torch.norm(v1,p=2).detach().clone()

    layer_number, layer_family = get_layer_number_and_family(param_name)
    percent_change = abs((numerators/denominators)* 100)

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


# Build a dict that stores the weight changes of params belonging to each family of param.
# e.g. {'query':{'module.1.query.weight' : 10, 'module.7.query.weight': 5}, 'key': {....}}
def get_family_param_weight_change(all_params, family_groups, window_count):
    family_params_weight_change = {}

    for param_family in family_groups:
        for param_name in family_groups[param_family]:
            _, family, pc = get_percent_change_in_norm(param_name, all_params[-2][param_name], all_params[-1][param_name])

            if family in family_params_weight_change:
                family_params_weight_change[family][param_name] = pc
            else:
                family_params_weight_change[family] = {}
                family_params_weight_change[family][param_name] = pc
    
    return family_params_weight_change

# Normalize the weight changes seperately for the param based on the family it belongs to.
# For example if a param belongs to 'query' family, obtain min and max from only the params in that family for normalization.
def get_family_wise_normalized_weights(family_params_weight_change):
    param_dict = {}

    for family in family_params_weight_change:
        all_weight_changes = torch.tensor(list(family_params_weight_change[family].values()))

        family_params_weights = family_params_weight_change[family]

        for param in family_params_weights:
            param_dict[param] = (family_params_weights[param] - all_weight_changes.min())/(all_weight_changes.max() - all_weight_changes.min())
    
    return param_dict

# Consider all the param name irrespective of the family they belong to
# Get max and min across all the params and perform normalization.
def get_overall_normalized_weights(all_params):
    all_param_names = all_params[0].keys()
    param_dict = {}

    for param_name in all_param_names:
        _, family, pc = get_percent_change_in_norm(param_name, all_params[-2][param_name], all_params[-1][param_name])

        param_dict[param_name] = pc
    
    max_weight = max(list(param_dict.values()))
    min_weight = min(list(param_dict.values()))

    for param_name in param_dict:
        param_dict[param_name] = (param_dict[param_name] - min_weight)/(max_weight - min_weight)

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

def get_selected_modules_weights(model, selected_modules):
    selected_modules_weights = {}

    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and is_one_of_module(name, selected_modules):
            if hasattr(module, 'weight'):
                selected_modules_weights[name] = module.weight.data.clone()
    
    return selected_modules_weights