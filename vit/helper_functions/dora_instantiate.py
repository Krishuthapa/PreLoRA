import re
import os

import torch
import torch.distributed as dist

from peft import get_peft_model, LoraConfig, TaskType

from helper_functions.distributed import print_at_master

from helper_functions.distributed import to_ddp
from helper_functions.schedulers import update_scheduler
from helper_functions.optimizers import update_optimizer
from helper_functions.rank_assignment_algorithm import rank_assignment_algorithm

def instantiate_dora_model(model, optimizer, scheduler, args, selected_modules, stored_epoch_metrics,iteration_counter):
    all_metrics_info  = [stored_epoch_metrics[iteration_num] for iteration_num in sorted(stored_epoch_metrics.keys())]
    modules_names = all_metrics_info[0].keys()

    assigned_ranks = rank_assignment_algorithm(all_metrics_info, args.low_rank,args.high_rank, 'cluster')

    assigned_rank_patterns = {}
    assigned_alpha_patterns = {}

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 0))


    for rank in assigned_ranks.keys():
        for module in assigned_ranks[rank]:
            assigned_rank_patterns[module] = int(rank)
            assigned_alpha_patterns[module] = int(rank) * args.lora_scaling
    
    peft_config = LoraConfig(
        inference_mode= False,
        target_modules=selected_modules,
        r=args.default_rank,
        rank_pattern = assigned_rank_patterns,
        alpha_pattern = assigned_rank_patterns,
        lora_alpha = args.default_rank * args.lora_scaling,
        lora_dropout = args.lora_dropout)
    
    peft_config_store = {
        'inference_mode': False,
        'target_modules': selected_modules,
        'default_r': args.default_rank,
        'rank_pattern': assigned_rank_patterns,
        'alpha_pattern': assigned_rank_patterns,
        'lora_alpha': args.default_rank * args.lora_scaling,
        'lora_dropout': args.lora_dropout
    }

    torch.save({'peft_config': peft_config_store},"/lus/grand/projects/datascience/kthapa/vit-lucidrain/checkpoint_dora/peft_config.pth")

    base_model = model.module
    model = get_peft_model(base_model,peft_config)
    model.print_trainable_parameters()

    for param in model.base_model.parameters():
        param.requires_grad = True

    model = to_ddp(model, args, local_rank)
    model.module.print_trainable_parameters()

    lora_targeted_modules = []
    for name, module in model.module.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            lora_targeted_modules.append(name)
    
    prev_lr = optimizer.param_groups[0]['lr']
    optimizer = update_optimizer(model, prev_lr, args)
    scheduler = update_scheduler(args,optimizer,scheduler, world_size, iteration_counter)

    return model, optimizer, scheduler, lora_targeted_modules

def freeze_base_layer_params(args, model, optimizer, scheduler, base_layers_to_freeze, iteration_counter):
    for name, module in model.module.named_modules():
        full_base_layer_name = name + '.base_layer'
        if full_base_layer_name in base_layers_to_freeze:
            if hasattr(module, 'base_layer'):
                already_frozen = all(not param.requires_grad for param in module.base_layer.parameters())
                if already_frozen:
                    print_at_master(f"Already frozen: {full_base_layer_name}")
                    continue
                for param in module.base_layer.parameters():
                    param.requires_grad = False
                print_at_master(f"Froze base_layer of: {full_base_layer_name}")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 0))

    base_model = model.module
    model = to_ddp(base_model, args, local_rank)

    prev_lr = optimizer.param_groups[0]['lr']
    optimizer = update_optimizer(model, prev_lr, args)
    scheduler = update_scheduler(args,optimizer,scheduler, world_size, iteration_counter)

    return model, optimizer, scheduler