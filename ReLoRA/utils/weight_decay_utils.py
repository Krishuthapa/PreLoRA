import torch

import numpy as np
import pandas as pd

import torch.nn as nn

# Power law implementation.
def powerlaw_tail_custom(num_layers, last_weight=0.5, alpha = 10):
    w = np.arange(1, num_layers) ** alpha
    w = (1-last_weight) * w / w.sum()

    assert w[-1] < last_weight, f"The second last layer cannot have higher weight than the last layer i.e. {round(w[-1],3)} > {last_weight}."

    return np.append(w, last_weight)

# exponential decay implementation.
def exponential_decay(num_layers, last_weight = 0.5, alpha = 0.5):
    w = 1 * np.exp(alpha * np.arange(1,num_layers))
    w = (1-last_weight) * w / w.sum()

    assert w[-1] < last_weight, f"The second last layer cannot have higher weight than the last layer i.e. {round(w[-1],3)} > {last_weight}."

    return np.append(w, last_weight)

# linear decay implementation.
def linear_decay(num_layers, last_weight = 0.5):
    slope = ((1- last_weight)-0)/(num_layers) 
    w = np.array([])

    for layer_num in range(1, num_layers):
        w = np.append(w,slope * layer_num)

    w = (1-last_weight) * w / w.sum()

    assert w[-1] < last_weight, f"The second last layer cannot have higher weight than the last layer i.e. {round(w[-1],3)} > {last_weight}."
        
    return np.append(w, last_weight)