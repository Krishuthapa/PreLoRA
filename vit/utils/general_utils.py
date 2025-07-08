import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from scipy import stats 

# Utility function to check if the given value is the power of 2.
def is_power_of_two(n):
  if n <= 0:
    return False
  return (n & (n - 1)) == 0

# Perform paired t-test (Null hypothesis => There is no significant difference in 2 lists.)
def is_t_test_passed(list1,list2):
  list1 = list1.numpy().tolist()
  list2 = list2.numpy().tolist()

  t_statistic, p_value = stats.ttest_rel(list1, list2)

  return p_value < 0.05 

def get_total_iteration_count(data_count, world_size, epochs, batch_size):
  num_iteration_per_epoch = (data_count // world_size) // batch_size

  return epochs * num_iteration_per_epoch
