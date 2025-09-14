import re
import os

import math

import torch
import torch.nn as nn
import torch.distributed as dist

from peft import get_peft_model, LoraConfig, TaskType

from helper_functions.distributed import print_at_master

@torch.no_grad()
def merge_and_reinit_model(model):
    for name, module in model.module.named_modules():
        if hasattr(module,"lora_A") and hasattr(module,"lora_B"):
            scaling = module.scaling['default']
            
            _lora_del_wt = module.lora_B['default'].weight @ module.lora_A['default'].weight * scaling
            module.base_layer.weight.data += _lora_del_wt

            nn.init.kaiming_uniform_(module.lora_A['default'].weight, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B['default'].weight)
    
    return model

@torch.no_grad()
def merge_and_reinit_model_old(model):
    model.module.merge_and_unload()
    model.module.enable_adapters()

    for name, module in model.module.named_modules():
        if hasattr(module,"lora_A") and hasattr(module,"lora_B"):
            A = module.lora_A['default'].weight
            B = module.lora_B['default'].weight

            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)
    
    return model