import re
import os

import torch
import torch.distributed as dist

from peft import get_peft_model, LoraConfig, TaskType

from helper_functions.distributed import print_at_master

from helper_functions.distributed import to_ddp
from helper_functions.schedulers import update_scheduler, WarmupOverrideScheduler
from helper_functions.optimizers import create_optimizer_adamW
from helper_functions.rank_assignment_algorithm import rank_assignment_algorithm


def toggle_base_model(model, scheduler, args, local_rank, total_images=1.28e6, freeze=False, lr_degrade = 1):
    base_model = model.module if hasattr(model, "module") else model

    for name, param in base_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = not freeze

    new_model = base_model.to(local_rank)
    new_model = to_ddp(new_model, args, local_rank)

    new_optimizer = create_optimizer_adamW(new_model, args.lr * lr_degrade, args)

    if hasattr(scheduler, 'original_scheduler'):
        current_scheduler = scheduler.original_scheduler
    else:
        current_scheduler = scheduler
    
    new_scheduler = update_scheduler(args, new_optimizer, current_scheduler, int(os.environ.get("WORLD_SIZE", 0)), total_images)
    new_scheduler = WarmupOverrideScheduler(new_optimizer, new_scheduler, warmup_steps=312)
    new_scheduler.start_warmup()

    return new_model, new_optimizer, new_scheduler


def instantiate_relora_model(model, optimizer, scheduler, args, selected_modules, 
                             stored_epoch_metrics, iteration_counter, total_images=1.28e6, use_dora=False):
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    all_metrics_info  = [stored_epoch_metrics[iteration_num] for iteration_num in sorted(stored_epoch_metrics.keys())]

    current_effective_iteration = int(iteration_counter / args.grad_accum_steps)

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
        'lora_alpha': args.default_rank * 64,
        'lora_dropout': args.lora_dropout,
        'use_dora': use_dora
    }
    torch.save({'peft_config': peft_config_store}, args.lora_cfg_pth_ac)

    if hasattr(model, "module"):
        base_model = model.module.merge_and_unload()
    else:
        base_model = model.merge_and_unload()

    if hasattr(scheduler,"original_scheduler"):
        scheduler = scheduler.original_scheduler
    
    new_model = get_peft_model(base_model, peft_config)
    new_model, new_optimizer, new_scheduler = toggle_base_model(new_model, scheduler, args, local_rank, total_images, False,1.0)

    lora_targeted_modules = [
        name for name, module in new_model.module.named_modules()
        if hasattr(module, "lora_A") or hasattr(module, "lora_B")
    ]

    del model, optimizer, scheduler, base_model
    torch.cuda.empty_cache()

    return new_model, new_optimizer, new_scheduler, lora_targeted_modules
