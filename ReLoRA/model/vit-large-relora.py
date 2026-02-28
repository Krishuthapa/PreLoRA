
import sys
import os

import math

import torch
import torch.optim
import torch.nn as nn

import torch.nn.parallel
import torch.distributed as dist
import torch.utils.data.distributed

from torch.amp import GradScaler, autocast

import time
import argparse

from peft import get_peft_model, LoraConfig, TaskType

from dataloader.dataloader import create_data_loaders
from helper_functions.general_helper_functions import accuracy, AverageMeter, silence_PIL_warnings, HookController
from helper_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib, get_dist_info

from helper_functions.schedulers import get_scheduler, WarmupOverrideScheduler

from helper_functions.optimizers import create_optimizer_adamW
from helper_functions.reset_optimizers import optimizer_reset
from helper_functions.merge_and_reinit import merge_and_reinit_model

from helper_functions.losses import SoftTargetCrossEntropy
from helper_functions.dora_instantiate import instantiate_relora_model
from helper_functions.convergence import check_partial_convergence, check_and_remove_unnecessary_metric

from peft import get_peft_model, LoraConfig, TaskType
from transformers import ViTConfig, ViTForImageClassification

from utils.initialize_vars import initialize_vars_relora
from utils.general_utils import get_total_iteration_count
from utils.model_utils import get_initialzed_model, save_model, get_peft_config, get_parsed_args
from utils.param_utils import get_selected_modules_norms, average_module_norms_across_gpus, get_selected_modules_lora_norms

import random
import numpy as np

import wandb
from timm.data import Mixup

parser = get_parsed_args(argparse)

def get_mixup(args):
    mixup_fn = Mixup(
    mixup_alpha=args.mixup_alpha,
    cutmix_alpha=args.cutmix_alpha,  
    cutmix_minmax=None,
    prob=args.mixup_prob,
    switch_prob=args.switch_prob,
    mode='batch',
    num_classes=args.num_classes
    )

    return mixup_fn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # arguments
    args = parser.parse_args()

    silence_PIL_warnings()

    setup_distrib(args)

    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))

    print_at_master("model : {} , patch: {} and img_size: {}".format(args.model_name, args.patch_size, args.img_size))
    print_at_master("nlayers: {}, dim : {} , heads: {} and mlp_dim: {}".format(args.nlayers, args.hidden_dim, args.nheads, args.ffn_dim))

    set_seed(100)

    model = get_initialzed_model(args,local_rank)
    
    if global_rank == 0:
        wandb.login()
        wandb.init(project=args.wandb_name, config = args)
        wandb.watch(model)

    train_loader, val_loader = create_data_loaders(args)

    train_21k(args, model, train_loader, val_loader, world_size, global_rank, local_rank)

def train_21k(args, model, train_loader, val_loader, world_size, global_rank, local_rank):
    optimizer_state_keys = ["exp_avg", "exp_avg_sq"]
    scaler = GradScaler(device='cuda')
    loss_fn = SoftTargetCrossEntropy()

    selected_modules = args.selected_modules.split(',')

    #############Checks for the checkpoint and if not assigns the defaults values to the variables.######################
    all_vars = initialize_vars_relora(args.checkpoint_path, model, args)
    
    model = all_vars[0]
    optimizer = all_vars[1]
    scheduler = all_vars[2]

    start_epoch = all_vars[3]
    iteration_counter = all_vars[6]

    stored_k1_weight_norms = all_vars[4]
    stored_k1_losses = all_vars[5]
    k1_total_loss = all_vars[7]

    is_model_warmed_up  = all_vars[8]
    has_convergence_passed = all_vars[9]
    #############################################################################################################

    model = to_ddp(model,args,local_rank)
    model.train()

    print_at_master(f"Here is the model: {model.module}")
    total_iteration = get_total_iteration_count(1.28e6, world_size, args.epochs, args.batch_size, args.grad_accum_steps)
    print_at_master(f"Iteration counter: {int(iteration_counter/args.grad_accum_steps)}")
    print_at_master(f"Total Iteration: {total_iteration}")

    trainable_params = [p for p in model.module.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.module.named_parameters() if p.requires_grad and "lora_" in n]
    trainable_params_names = [name for name, p in model.module.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'] if p.requires_grad)
    print_at_master(f"Trainable parameters according to optimizer: {total_params}")
    
    if len(lora_params) > 0 and local_rank == 0:
        model.module.print_trainable_parameters()
        
    # training loop
    for epoch in range(start_epoch, args.epochs):
        if num_distrib() > 1:
            train_loader.sampler.set_epoch(epoch)
       
        epoch_start_time = time.time()

        total_loss = 0.0
        num_batches = 0

        correct_top1 = torch.tensor(0.0, device=f"cuda:{local_rank}")
        correct_top5 = torch.tensor(0.0, device=f"cuda:{local_rank}")
        total_samples = torch.tensor(0.0, device=f"cuda:{local_rank}")

        for i, (input, target) in enumerate(train_loader):
            input = input.to(local_rank, non_blocking=True)
            target = target.to(local_rank, non_blocking=True)
            
            if bool(args.has_mixup):
                mixup_fn = get_mixup(args)
                input,target = mixup_fn(input,target)
            
            with autocast(device_type='cuda'):  # mixed precision
                output = model(input)
                loss = loss_fn(output.logits, target)  # note - loss also in fp16
                
            scaler.scale(loss).backward()

            if (i + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            loss_collector = loss.detach().clone()

            if num_distrib() > 1:
                dist.all_reduce(loss_collector, op=dist.ReduceOp.SUM)
            
            avg_loss_acc_gpus = loss_collector / world_size

            total_loss += avg_loss_acc_gpus.item()
            k1_total_loss += avg_loss_acc_gpus.item()
            num_batches += 1

            with torch.no_grad():
                maxk = 5
                _, pred = output.logits.topk(maxk, 1, True, True)
                pred = pred.t()

                hard_target = target.argmax(dim=1)
                correct = pred.eq(hard_target.view(1, -1).expand_as(pred))
            
                correct_top1 += correct[:1].reshape(-1).float().sum()
                correct_top5 += correct[:5].reshape(-1).float().sum()
                total_samples += target.size(0)

            ############################################## Switching to ReLoRA #################################################
            current_effective_iteration = int(iteration_counter / args.grad_accum_steps)
            

            if int(current_effective_iteration) == int(args.relora_switch_step) and not args.has_lora:
                peft_config, peft_config_store = get_peft_config(selected_modules, args)
                
                is_model_warmed_up = True
                torch.save({'peft_config': peft_config_store}, args.lora_cfg_pth_bc)

                base_model = model.module
                peft_model = get_peft_model(base_model,peft_config)
                peft_model = peft_model.to(local_rank)

                new_optimizer = create_optimizer_adamW(model, args.lr, args)
                new_scheduler = get_scheduler(args, new_optimizer, world_size, 1.28e6)
                new_scheduler.step(current_effective_iteration)
                
                scaler = GradScaler(device='cuda')

                for module in model.modules():
                    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                            A = module.lora_A['default'].weight
                            B = module.lora_B['default'].weight
                            
                            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                            nn.init.zeros_(B)

                peft_model = to_ddp(peft_model,args,local_rank)
                
                if global_rank == 0:
                    save_model(args, peft_model, optimizer, scheduler, epoch, iteration_counter, has_convergence_passed, is_model_warmed_up, stored_k1_losses, stored_k1_weight_norms, k1_total_loss)
                    
                    print_at_master(f"After model warmup done:{epoch}")
                    peft_model.module.print_trainable_parameters()
                
                model= peft_model
                optimizer= new_optimizer
                scheduler = new_scheduler

                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            
            ############################################## LoRA check and implementation ########################################
            should_switch = torch.tensor([0], device=f"cuda:{local_rank}")

            if is_model_warmed_up and (int(current_effective_iteration) - args.relora_switch_step) > 0 and (int(current_effective_iteration) - args.relora_switch_step) % args.k1_steps == 0 and (i + 1) % args.grad_accum_steps == 0:
                print_at_master("2: ya ho")
                if not has_convergence_passed:
                    k1_selected_weight_norms = get_selected_modules_lora_norms(model, selected_modules)
                    k1_selected_weight_norms = average_module_norms_across_gpus(k1_selected_weight_norms, local_rank)

                    stored_k1_weight_norms[iteration_counter] = k1_selected_weight_norms
                    stored_k1_losses[iteration_counter] = k1_total_loss/args.k1_steps

                    k1_total_loss = 0.0

                if not has_convergence_passed and len(stored_k1_weight_norms.keys()) >= int(args.consecutive_windows):
                    stored_k1_losses = check_and_remove_unnecessary_metric(stored_k1_losses, args.consecutive_windows)
                    stored_k1_weight_norms = check_and_remove_unnecessary_metric(stored_k1_weight_norms, args.consecutive_windows)

                    if global_rank == 0:                
                        is_partially_converged = check_partial_convergence(args,stored_k1_losses, stored_k1_weight_norms, thr_loss= args.k1_loss_thr, thr_norms = args.k1_wc_thr, checking_step = args.k1_steps, checking_window = args.consecutive_windows)
                        should_switch[0] = 1 if is_partially_converged else 0

                if not has_convergence_passed:
                    dist.broadcast(should_switch, src=0)

            if is_model_warmed_up and (int(current_effective_iteration) - args.relora_switch_step) > 0 and (int(current_effective_iteration) - args.relora_switch_step) % args.merge_and_reinit_step == 0 and (i + 1) % args.grad_accum_steps == 0:
                print_at_master(f"Performing lora reset at update step {current_effective_iteration}. Before lr is {optimizer.param_groups[0]['lr']}")
                
                lora_reset_start_time = time.time()
                model = merge_and_reinit_model(model)
                lora_reset_end_time = time.time()

                print_at_master(f"Total merge and reinit model time: {lora_reset_end_time - lora_reset_start_time}")

                lora_params = [p for n, p in model.module.named_parameters() if p.requires_grad and "lora_" in n]
                
                optimizer_reset(optimizer, lora_params, optimizer_state_keys, 0.75)
                new_scheduler = WarmupOverrideScheduler(optimizer, scheduler, warmup_steps=200)
                scheduler = new_scheduler
                scheduler.start_warmup()

                scaler = GradScaler(device='cuda')

                torch.cuda.empty_cache()    
                print_at_master(f"Performing lora reset at update step {current_effective_iteration}. After lr is {optimizer.param_groups[0]['lr']}")

            if should_switch.item() == 1 and not has_convergence_passed:                
                dora_initialize_start = time.time()
                model, optimizer, scheduler, targeted_lora_modules = instantiate_relora_model(model, optimizer, scheduler, args, selected_modules, stored_k1_weight_norms, iteration_counter, use_dora = bool(args.use_dora))
                dora_initialize_end = time.time()

                scaler = GradScaler(device='cuda')

                has_convergence_passed = True
                
                if global_rank == 0:
                    
                    save_model(args, model, optimizer, scheduler, epoch, iteration_counter, has_convergence_passed, is_model_warmed_up, stored_k1_losses, stored_k1_weight_norms, k1_total_loss)
                    
                    print_at_master(f"After rank assigned lora implemented.: {epoch}")
                    model.module.print_trainable_parameters()
                
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            #################################################################### LoRA implementation ends #############################################################
            if (i + 1) % args.grad_accum_steps == 0:
                optimizer.zero_grad()
            
            iteration_counter += 1
            del input, target

        if (i + 1) % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start_time
        print_at_master(f"Iteration counter: {int(iteration_counter/args.grad_accum_steps)}")
        print_at_master(f"Total Iteration: {total_iteration}")

        dist.all_reduce(correct_top1, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_top5, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        train_top1_acc = 100.0 * correct_top1.item() / total_samples.item()
        train_top5_acc = 100.0 * correct_top5.item() / total_samples.item()

        if global_rank == 0 and has_convergence_passed:
            print_at_master("before validation")
            model.module.print_trainable_parameters()

        #validation epoch
        top1_loss, top5_loss = validate_21k(val_loader, model)

        if global_rank == 0:
            wandb.log({'epoch': epoch, 
                'loss':total_loss/num_batches, 
                'lr': optimizer.param_groups[0]['lr'],
                'top1_val_acc': top1_loss,
                'top5_val_acc':top5_loss,
                'training_rate [img/sec]': len(train_loader) * args.batch_size / epoch_time * max(num_distrib(),1),
                'epoch_time': epoch_time,
                'train-top1-acc': train_top1_acc,
                'train-top5-acc': train_top5_acc })

        if ((epoch> 0 and epoch% 5 == 0) or epoch == (args.epochs-1)) and global_rank == 0:
            save_model(args, model, optimizer, scheduler, epoch, iteration_counter, has_convergence_passed, is_model_warmed_up, stored_k1_losses, stored_k1_weight_norms, k1_total_loss)

        model.train()

@torch.no_grad()
def validate_21k(val_loader, model):
    model.eval()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    top1_sum = 0.0
    top5_sum = 0.0
    total = 0

    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            outputs = model(inputs)
            logits = outputs.logits.float()  # adjust if model does not use `.logits`

        # Compute top-1 and top-5
        maxk = 5
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        correct_top1 = correct[:1].reshape(-1).float().sum(0)
        correct_top5 = correct[:5].reshape(-1).float().sum(0)
        batch_size = torch.tensor(targets.size(0), device=device, dtype=torch.float32)

        # Reduce across GPUs
        dist.all_reduce(correct_top1, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_top5, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_size, op=dist.ReduceOp.SUM)

        top1_sum += correct_top1.item()
        top5_sum += correct_top5.item()
        total += batch_size.item()

    # Final accuracy
    top1_avg = 100.0 * top1_sum / total
    top5_avg = 100.0 * top5_sum / total

    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Validation Top-1 Accuracy: {top1_avg:.2f}%")
        print(f"Validation Top-5 Accuracy: {top5_avg:.2f}%")

    return top1_avg, top5_avg

if __name__ == '__main__':
    main()