import os
import torch
import shutil

import math

import torch
import torch.optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

from helper_functions.distributed import print_at_master

def get_scheduler(args, optimizer, world_size, total_images=1.28e6):
    steps_per_epoch = int(total_images // (args.batch_size * world_size * args.grad_accum_steps))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps

    def lr_lambda(current_step):
        if current_step <= warmup_steps:
            return float(current_step) / warmup_steps
        
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler
    

def update_scheduler(args, optimizer, scheduler, world_size, total_images=1.28e6):
    current_step = scheduler.last_epoch - 1

    scheduler = get_scheduler(args,optimizer, world_size, total_images)
    scheduler.step(current_step + 1)

    return scheduler


class WarmupOverrideScheduler:
    def __init__(self, optimizer, original_scheduler, warmup_steps=50):
        self.optimizer = optimizer
        self.original_scheduler = original_scheduler
        self.warmup_steps = warmup_steps
        self.active = False
        self.start_lrs = None
        self.step_count = 0

    def start_warmup(self):
        # Record current LR as target
        self.start_lrs = [g['lr'] for g in self.optimizer.param_groups]
        # Set LR to 0 initially
        for g in self.optimizer.param_groups:
            g['lr'] = 0.0
        self.active = True
        self.step_count = 0

    def step(self):
        if self.active:
            self.step_count += 1
            alpha = self.step_count / self.warmup_steps
            if alpha >= 1.0:
                self.active = False
                # resume original scheduler (it will step as usual)
            else:
                # Linear warmup from 0 → start_lrs
                for lr, g in zip(self.start_lrs, self.optimizer.param_groups):
                    g['lr'] = lr * alpha
        else:
            self.original_scheduler.step()
