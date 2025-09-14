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

def scheduler_with_short_warmup_old(
    args, optimizer, previous_scheduler, world_size,
    total_images=1.28e6,
    warmup_steps_after_freeze =100,
):
    last_iter = previous_scheduler.last_epoch

    steps_per_epoch = int(total_images // (args.batch_size * world_size * args.grad_accum_steps))
    total_steps = args.epochs * steps_per_epoch
    
    warmup_steps = warmup_steps_after_freeze + last_iter

    def lr_lambda(current_step):
        progress = float(current_step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)

        if current_step <= warmup_steps:
            return (float(current_step - last_iter) / warmup_steps_after_freeze) * (0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler.step(last_iter + 1)

    return scheduler


def scheduler_with_short_warmup(
    args, optimizer, previous_scheduler, world_size,
    total_images=1.28e6,
    warmup_steps_after_freeze=100,
):
    last_iter = previous_scheduler.last_epoch
    
    steps_per_epoch = int(total_images // (args.batch_size * world_size * args.grad_accum_steps))
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(current_step):
        print_at_master(f"{last_iter},{current_step}")

        rel_step = current_step - last_iter  

        if rel_step <= warmup_steps_after_freeze:
            return float(rel_step) / float(warmup_steps_after_freeze)
        
        progress = float(current_step) / float(total_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler.step(last_iter + 1)
    return scheduler


def update_scheduler_with_decay(args, optimizer, scheduler, world_size, total_images=1.28e6, decay_speedup=2.5):
    current_step = scheduler.last_epoch - 1  # Continue from current

    steps_per_epoch = int(total_images // (args.batch_size * world_size * args.grad_accum_steps))
    
    # Original total steps
    original_total_steps = args.epochs * steps_per_epoch

    # Shorten decay horizon
    remaining_steps = max(original_total_steps - current_step, 1)
    new_total_steps = int(remaining_steps / decay_speedup)

    warmup_steps = args.warmup_steps 

    def lr_lambda(step):
        if step <= warmup_steps:
            return float(step) / warmup_steps
        progress = float(step - warmup_steps) / max(1, new_total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler.step(current_step + 1)

    return scheduler

    