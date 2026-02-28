import os

import torch

from helper_functions.distributed import print_at_master, to_ddp

from helper_functions.schedulers import get_scheduler
from helper_functions.optimizers import create_optimizer_adamW
from helper_functions.general_helper_functions import check_container_and_assign

def initialize_vars_relora(checkpoint_path, model, args, total_images=1.28e6):
    world_size = int(os.environ.get('WORLD_SIZE'))
    local_rank = int(os.environ.get('LOCAL_RANK'))

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model from checkpoint knowing which parameters were frozen and trainable.
        model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('trainable_params', None) is not None:
            for name, param in model.named_parameters():
                if name in checkpoint["trainable_params"]:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        learning_rate = check_container_and_assign(checkpoint,'learning_rate', 0.0)
        iteration_counter = check_container_and_assign(checkpoint,'iteration_counter',checkpoint['epoch'] * int(total_images/ (args.batch_size * world_size * args.grad_accum_steps)))        
        is_model_warmed_up = check_container_and_assign(checkpoint,'is_model_warmup_done', False)
        is_conv_warmed_up = check_container_and_assign(checkpoint,'is_conv_warmup_done', False)

        has_convergence_passed = check_container_and_assign(checkpoint,'has_convergence_passed', False)

        start_epoch = checkpoint['epoch'] + 1
        switch_model_start_iter = check_container_and_assign(checkpoint,'switch_model_start_iter', None)

        stored_k1_weight_norms = check_container_and_assign(checkpoint, 'stored_k1_weight_norms', {})
        stored_k1_losses = check_container_and_assign(checkpoint, 'stored_k1_losses', {})
        k1_total_loss = check_container_and_assign(checkpoint, 'k1_total_loss', 0.0)

        optimizer = create_optimizer_adamW(model, args.lr, args)
        scheduler = get_scheduler(args, optimizer, world_size, 1.28e6)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print_at_master(f"Learning rate intitialized: {learning_rate}")

    else:
        start_epoch = 0

        stored_k1_weight_norms = {}
        stored_k1_losses = {}

        iteration_counter = 0
        k1_total_loss = 0.0

        is_model_warmed_up = False
        has_convergence_passed = False
        is_conv_warmed_up = False

        switch_model_start_iter = None

        optimizer = create_optimizer_adamW(model, args.lr, args)
        scheduler = get_scheduler(args, optimizer, world_size, 1.28e6)

    
    return (model, optimizer, scheduler, start_epoch, stored_k1_weight_norms, stored_k1_losses, iteration_counter, k1_total_loss , is_model_warmed_up, is_conv_warmed_up, has_convergence_passed, switch_model_start_iter)