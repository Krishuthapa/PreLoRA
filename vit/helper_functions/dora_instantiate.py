from peft import get_peft_model, LoraConfig, TaskType
import torch
import re

import torch.distributed as dist

from helper_functions.rank_assignment_algorithm import rank_assignment_algorithm

def instantiate_dora_model(model, args, selected_modules, stored_epoch_weights):
    all_weights_info  = [stored_epoch_weights[iteration_num] for iteration_num in sorted(stored_epoch_weights.keys())]
    modules_names = all_weights_info[0].keys()

    assigned_ranks = rank_assignment_algorithm(all_weights_info, args.low_rank,args.high_rank)

    assigned_rank_patterns = {}
    assigned_alpha_patterns = {}

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

    model.get_peft_model(model,peft_config)
    model.print_trainable_parameters()

    for param in model.base_model.parameters():
        param.requires_grad = True

    lora_targeted_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            lora_targeted_modules.append(name)


    return model, lora_targeted_modules