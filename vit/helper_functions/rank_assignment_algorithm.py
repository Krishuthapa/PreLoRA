import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from utils.general_utils import is_power_of_two
from utils.param_utils import cluster_param_based_on_family, get_family_param_weight_change, get_family_wise_normalized_weights, get_overall_normalized_weights

def rank_assignment_algorithm(all_params, min_rank = 4, max_rank = 32, normalized_as='overall'):
    all_param_names = all_params[0].keys()

    normalized_weight_changes = {}
    
    assert is_power_of_two(min_rank), "min rank should be a number power of 2."
    assert is_power_of_two(max_rank), "max rank should be a number power of 2." 
    assert min_rank < max_rank, "min rank should be smaller than max rank."

    if normalized_as == 'cluster':
        family_param_cluster = cluster_param_based_on_family(all_param_names)
        family_params_weight_change = get_family_param_weight_change(all_params, family_param_cluster)

        normalized_weight_changes = get_family_wise_normalized_weights(family_params_weight_change)
    else:
        normalized_weight_changes = get_overall_normalized_weights(all_params)

    start_rank_power = int(math.log2(min_rank))
    end_rank_power = int(math.log2(max_rank))

    all_ranks = []
    assigned_ranks = {}
    for power in range(start_rank_power, end_rank_power+1):
        assigned_ranks[2**power] = []
        all_ranks.append(2**power)
    
    for key in normalized_weight_changes.keys():
        rank_index = math.ceil(normalized_weight_changes[key] * len(all_ranks)) -1 if normalized_weight_changes[key] != 0 else math.ceil(normalized_weight_changes[key] * len(all_ranks))
        assigned_ranks[all_ranks[rank_index]].append(key)
    
    return assigned_ranks
