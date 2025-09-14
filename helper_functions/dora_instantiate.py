import re
import os

import torch
import torch.distributed as dist

from peft import get_peft_model, LoraConfig, TaskType

from helper_functions.distributed import print_at_master

from helper_functions.distributed import to_ddp
from helper_functions.schedulers import update_scheduler, update_scheduler_with_decay
from helper_functions.optimizers import update_optimizer
from helper_functions.rank_assignment_algorithm import rank_assignment_algorithm

def instantiate_dora_model(model, optimizer, scheduler, args, selected_modules, stored_epoch_metrics, iteration_counter, total_images=1.28e6, use_dora= False):
    all_metrics_info  = [stored_epoch_metrics[iteration_num] for iteration_num in sorted(stored_epoch_metrics.keys())]
    modules_names = all_metrics_info[0].keys()

    assigned_ranks = rank_assignment_algorithm(all_metrics_info, args.low_rank,args.high_rank, 'cluster')

    assigned_rank_patterns = {}
    assigned_alpha_patterns = {}

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 0))
    global_rank = int(os.environ.get("RANK", 0))

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
        lora_dropout = args.lora_dropout,
        use_dora=use_dora)
    
    peft_config_store = {
        'inference_mode': False,
        'target_modules': selected_modules,
        'default_r': args.default_rank,
        'rank_pattern': assigned_rank_patterns,
        'alpha_pattern': assigned_rank_patterns,
        'lora_alpha': args.default_rank * args.lora_scaling,
        'lora_dropout': args.lora_dropout,
        'use_dora':use_dora
    }

    torch.save({'peft_config': peft_config_store},args.lora_cfg_pth)

    base_model = model.module
    model = get_peft_model(base_model,peft_config)

    if global_rank == 0:
        model.print_trainable_parameters()

    for param in model.base_model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent_params = trainable_params / total_params

    optimizer = update_optimizer(model, args.lr, args)
    scheduler = update_scheduler(args,optimizer,scheduler, world_size, total_images)

    model = to_ddp(model, args, local_rank)

    if global_rank == 0:
        model.module.print_trainable_parameters()

    lora_targeted_modules = []
    for name, module in model.module.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            lora_targeted_modules.append(name)
    
    return model, optimizer, scheduler, lora_targeted_modules

def freeze_base_layer_params(args, model, base_layers_to_freeze):
    base_model = model.module

    for name, module in base_model.named_modules():
        full_base_layer_name = name + '.base_layer'
        if full_base_layer_name in base_layers_to_freeze:
            if hasattr(module, 'base_layer'):
                already_frozen = all(not param.requires_grad for param in module.base_layer.parameters())
                if already_frozen:
                    continue
                for param in module.base_layer.parameters():
                    param.requires_grad = False
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 0))

    model = to_ddp(base_model, args, local_rank)

    return model

def freeze_all_base_layers(args, model):
    base_model = model.module

    for name, module in base_model.named_modules():
        if hasattr(module, 'base_layer'):
            for param in module.base_layer.parameters():
                param.requires_grad = False
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 0))

    model = to_ddp(base_model, args, local_rank)

    return model


def instantiate_relora_model(model, optimizer, scheduler, args, selected_modules, 
                             stored_epoch_metrics, iteration_counter, total_images=1.28e6, use_dora=False):
    all_metrics_info  = [stored_epoch_metrics[iteration_num] for iteration_num in sorted(stored_epoch_metrics.keys())]
    modules_names = all_metrics_info[0].keys()

    assigned_ranks = rank_assignment_algorithm(all_metrics_info, args.low_rank, args.high_rank, 'cluster')

    assigned_rank_patterns = {}
    assigned_alpha_patterns = {}

    for rank in assigned_ranks.keys():
        for module in assigned_ranks[rank]:
            assigned_rank_patterns[module] = int(rank)
            assigned_alpha_patterns[module] = int(rank) * args.lora_scaling

    peft_config = LoraConfig(
        inference_mode=False,
        target_modules=selected_modules,
        r=args.default_rank,
        rank_pattern=assigned_rank_patterns,
        alpha_pattern=assigned_alpha_patterns,
        lora_alpha=args.default_rank * args.lora_scaling,
        lora_dropout=args.lora_dropout,
        use_dora=use_dora
    )

    peft_config_store = {
        'inference_mode': False,
        'target_modules': selected_modules,
        'default_r': args.default_rank,
        'rank_pattern': assigned_rank_patterns,
        'alpha_pattern': assigned_alpha_patterns,
        'lora_alpha': args.default_rank * args.lora_scaling,
        'lora_dropout': args.lora_dropout,
        'use_dora': use_dora
    }
    torch.save({'peft_config': peft_config_store}, args.lora_cfg_pth)

    if hasattr(model, "module"):
        base_model = model.module.merge_and_unload()
    else:
        base_model = model.merge_and_unload()

    model = get_peft_model(base_model, peft_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent_params = trainable_params / total_params

    optimizer = update_optimizer(model, args.lr, args)
    scheduler = update_scheduler(args, optimizer, scheduler, int(os.environ.get("WORLD_SIZE", 0)), total_images)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    model = to_ddp(model, args, local_rank)

    if global_rank == 0:
        model.module.print_trainable_parameters()

    lora_targeted_modules = [
        name for name, module in model.module.named_modules()
        if hasattr(module, "lora_A") or hasattr(module, "lora_B")
    ]

    return model, optimizer, scheduler, lora_targeted_modules
