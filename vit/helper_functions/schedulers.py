import os
import torch
import shutil

import math

import torch
import torch.optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

def get_scheduler(args, optimizer, world_size):
    steps_per_epoch = int(1.28e6 // (args.batch_size * world_size))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps

    def lr_lambda(current_step):
        if current_step <= warmup_steps:
            return float(current_step) / warmup_steps
        
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

def update_scheduler(args, optimizer, scheduler, world_size, current_step=None):
    scheduler = get_scheduler(args, optimizer, world_size)
    
    if current_step is not None:
        scheduler.last_epoch = current_step - 1
        scheduler.step()
    else:
        scheduler.last_epoch = -1
    return scheduler