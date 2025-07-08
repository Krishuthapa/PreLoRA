import os

import torch

from helper_functions.distributed import print_at_master, to_ddp
from helper_functions.schedulers import update_scheduler
from helper_functions.optimizers import update_optimizer
from helper_functions.general_helper_functions import check_container_and_assign

def initialize_vars_dora(checkpoint_path, model, optimizer, scheduler, scaler, args, alternate=True):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        model.module.load_state_dict(checkpoint['model_state_dict'])

        world_size = int(os.environ.get('WORLD_SIZE'))
        local_rank = int(os.environ.get('LOCAL_RANK'))

        base_model = model.module
        model = to_ddp(base_model, args, local_rank)

        if checkpoint.get('trainable_params', None) is not None:
            for name, param in model.module.named_parameters():
                if name in checkpoint["trainable_params"]:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        learning_rate = check_container_and_assign(checkpoint,'learning_rate', 0.0)
        iteration_counter = check_container_and_assign(checkpoint,'iteration_counter',checkpoint['epoch'] * int(1.28e6/ (args.batch_size * 64)))

        if alternate:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            optimizer = update_optimizer(model, learning_rate, args)
            scheduler = update_scheduler(args, optimizer, scheduler, world_size, iteration_counter)
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        val_top1_losses = checkpoint['val_top1_losses'].tolist()
        val_top5_losses = checkpoint['val_top5_losses'].tolist()
        epoch_losses = checkpoint['epoch_losses'].tolist()

        stored_k1_weight_norms = check_container_and_assign(checkpoint, 'stored_k1_weight_norms', {})
        stored_k1_grad_norms = check_container_and_assign(checkpoint, 'stored_k1_grad_norms', {})
        stored_k1_losses = check_container_and_assign(checkpoint, 'stored_k1_losses', {})
        is_dora_initialized = check_container_and_assign(checkpoint,'is_dora_initialized', False)
        is_frozen = check_container_and_assign(checkpoint,'is_frozen', False)
        targeted_lora_parent_modules = check_container_and_assign(checkpoint,'targeted_lora_parent_modules',[])
        k1_total_loss = check_container_and_assign(checkpoint, 'k1_total_loss', 0.0)
    else:

        start_epoch = 0
        val_top1_losses = []
        val_top5_losses = []
        epoch_losses = []

        stored_k1_weight_norms = {}
        stored_k1_grad_norms = {}
        stored_k1_losses = {}
        targeted_lora_parent_modules = []
        is_dora_initialized = False
        is_frozen = False
        iteration_counter = 0
        k1_total_loss = 0.0
    
    return (model, optimizer, scheduler, scaler, start_epoch, val_top1_losses, val_top5_losses, epoch_losses, stored_k1_weight_norms, stored_k1_grad_norms, stored_k1_losses, targeted_lora_parent_modules, is_dora_initialized, is_frozen, iteration_counter, k1_total_loss)