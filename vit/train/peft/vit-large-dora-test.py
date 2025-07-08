
import sys
import os

import math

import torch
import torch.optim

import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data.distributed

from torch.amp import GradScaler, autocast

import time
import argparse

from peft import get_peft_model, LoraConfig, TaskType

from dataloader.data_loaderi1k import create_data_loaders
from helper_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib, get_dist_info
from helper_functions.general_helper_functions import accuracy, AverageMeter, silence_PIL_warnings

from helper_functions.optimizers import create_optimizer_adam, update_optimizer
from helper_functions.schedulers import get_scheduler, update_scheduler
from helper_functions.losses import CrossEntropyLS, SoftTargetCrossEntropy
from helper_functions.convergence import check_partial_convergence, check_and_remove_unnecessary_metric, check_activation_change
from helper_functions.dora_instantiate import instantiate_dora_model

from transformers import ViTConfig, ViTForImageClassification

from utils.param_utils import get_selected_modules_norms, average_module_norms_across_gpus
from utils.general_utils import get_total_iteration_count
from utils.initialize_vars import initialize_vars_dora

import random
import numpy as np

import wandb
from timm.data import Mixup

parser = argparse.ArgumentParser(description='PyTorch ImageNet21K Single-label Training From Random Initialization')
parser.add_argument('--data_path', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--num_classes', default=1000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=140, type=int)
parser.add_argument('--warmup_steps', default=140, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument("--label_smooth", default=0.2, type=float)
parser.add_argument("--nlayers", default=24, type=int)
parser.add_argument("--hidden_dim", default=1024, type=int)
parser.add_argument("--ffn_dim", default=4096, type=int)
parser.add_argument("--nheads", default=16, type=int)
parser.add_argument("--patch_size", default=16, type=int)
parser.add_argument("--img_size", default=224, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--drop_path_rate", default=0.0, type=float)
parser.add_argument("--wandb_name", default='vit-hf', type=str)
parser.add_argument("--model_name", default='LARGE', type=str)


# Mixup / CutMix
parser.add_argument('--has_mixup', default = 1, type= int)
parser.add_argument('--mixup_alpha', default=0.0, type=float)
parser.add_argument('--cutmix_alpha', default=0.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)
parser.add_argument('--switch_prob', default=0.0, type=float)

# RandAugment
parser.add_argument('--randaugment_num_ops', default=2, type=int)
parser.add_argument('--randaugment_magnitude', default=9, type=int)

## DoRA arguments
parser.add_argument("--consecutive_windows", default=3,type=int)
parser.add_argument("--k2_consecutive_windows", default=3,type=int)
parser.add_argument("--k1_steps", default=1000,type=int)
parser.add_argument("--k1_loss_thr", default=10.0,type=float)
parser.add_argument("--k1_wc_thr", default=5.0,type=float)
parser.add_argument("--k2_steps", default=100,type=int)
parser.add_argument("--k2_freeze_thr", default=10, type=float)
parser.add_argument("--lora_scaling", default=2, type=int)
parser.add_argument("--lora_dropout", default=0.0, type=float)
parser.add_argument("--low_rank", default=4, type=int)
parser.add_argument("--high_rank", default=32, type=int)
parser.add_argument("--default_rank", default=8, type=int)
parser.add_argument("--has_lora", default=0, type=int)
parser.add_argument("--lora_cfg_pth", type=str)
parser.add_argument("--selected_modules", default="attention.query,attention.value", type=str)

def get_mixup(args):
    mixup_fn = Mixup(
    mixup_alpha=args.mixup_alpha,   # Mixup strength
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

def get_initialzed_model(args, local_rank):
    model_config = ViTConfig(hidden_size = args.hidden_dim, 
                    intermediate_size = args.ffn_dim,
                    image_size = args.img_size,
                    patch_size = args.patch_size,
                    num_attention_heads = args.nheads,
                    num_hidden_layers = args.nlayers,
                    num_labels = args.num_classes,
                    hidden_dropout_prob = args.dropout)
    
    if args.drop_path_rate > 0:
        model_config.update({"drop_path_rate": args.drop_path_rate})

    model = ViTForImageClassification(model_config).to(local_rank)

    if bool(args.has_lora):
        if os.path.exists(args.lora_cfg_pth):
            lora_config_checkpoint = torch.load(args.lora_cfg_pth, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            
            lora_config = lora_config_checkpoint['peft_config']
            
            peft_config = LoraConfig(
                    inference_mode= lora_config['inference_mode'],
                    target_modules=lora_config['target_modules'],
                    r=lora_config.get('default_r', None),
                    rank_pattern = lora_config.get('rank_pattern', None),
                    alpha_pattern = lora_config.get('alpha_pattern', None),
                    lora_alpha = lora_config.get('lora_alpha', None),
                    lora_dropout = lora_config.get('lora_dropout', None))
            
            model = get_peft_model(model,peft_config)

    model = to_ddp(model,args,local_rank)

    return model

def save_model(args, model, optimizer, scheduler, epoch, scaler, val_top1_losses, val_top5_losses, epoch_losses,is_dora_initialized, is_frozen, 
               targeted_lora_parent_modules, k1_total_loss, iteration_counter, stored_k1_weight_norms,stored_k1_grad_norms, stored_k1_losses):
    
    trainable_params = [name for name, param in model.module.named_parameters() if param.requires_grad]

    print_at_master(f" When model saved Optimizer learning rate: {optimizer.param_groups[0]['lr']} and epoch: {epoch}")

    
    torch.save({
            'model_state_dict':model.module.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'trainable_params': trainable_params,
            'epoch':epoch,
            'scaler':scaler.state_dict(),
            'val_top1_losses': torch.tensor(val_top1_losses),
            'val_top5_losses': torch.tensor(val_top5_losses),
            'epoch_losses': torch.tensor(epoch_losses),
            'is_dora_initialized': is_dora_initialized,
            'is_frozen':is_frozen,
            'targeted_lora_parent_modules':targeted_lora_parent_modules,
            'k1_total_loss':k1_total_loss,
            'iteration_counter': iteration_counter,
            'stored_k1_weight_norms': stored_k1_weight_norms,
            'stored_k1_grad_norms':stored_k1_grad_norms,
            'stored_k1_losses':stored_k1_losses,
            'learning_rate': optimizer.param_groups[0]['lr'] 
    },"/lus/grand/projects/datascience/kthapa/vit-lucidrain/checkpoint_dora/vit_checkpoint_dora_{}_{}.pth".format(args.model_name,epoch))

def main():
    # arguments
    args = parser.parse_args()

    silence_PIL_warnings()

    setup_distrib(args)

    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))

    print_at_master("model : {} , patch: {} and img_size: {}".format(args.model_name, args.patch_size, args.img_size))
    print_at_master("nlayers: {}, dim : {} , heads: {} and mlp_dim: {}".format(args.nlayers, args.hidden_dim, args.nheads, args.ffn_dim))

    set_seed(100)

    model = get_initialzed_model(args,local_rank)
    
    if global_rank == 0:
        wandb.login()
        wandb.init(project=args.wandb_name, config = args)
        wandb.watch(model)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.999))
    train_loader, val_loader = create_data_loaders(args)

    model.train()
    train_21k(model, train_loader, val_loader, optimizer, args)

def train_21k(model, train_loader, val_loader, optimizer, args):
    # set loss
    loss_fn = SoftTargetCrossEntropy()

    world_size = int(os.environ.get('WORLD_SIZE'))
    global_rank = int(os.environ.get('RANK'))
    local_rank = int(os.environ.get('LOCAL_RANK'))

    # set scheduler
    scheduler = get_scheduler(args, optimizer, world_size)
    
    # set scalaer
    scaler = GradScaler(device='cuda')

    # DoRA parameters.
    has_hooks= False
    selected_modules = args.selected_modules.split(',')
    is_stabilizing = False
    stabilizing_steps_remaining = 0

    #############Checks for the checkpoint and if not assigns the defaults values to the variables.##############
    all_vars = initialize_vars_dora(args.checkpoint_path, model, optimizer, scheduler, scaler, args, alternate = True)
    
    model = all_vars[0]
    optimizer = all_vars[1]
    scheduler = all_vars[2]
    scaler = all_vars[3]
    start_epoch = all_vars[4]
    val_top1_losses = all_vars[5]
    val_top5_losses = all_vars[6]
    epoch_losses = all_vars[7]
    stored_k1_weight_norms = all_vars[8]
    stored_k1_grad_norms = all_vars[9]
    stored_k1_losses = all_vars[10]
    targeted_lora_parent_modules = all_vars[11]
    is_dora_initialized = all_vars[12]
    is_frozen = all_vars[13]
    iteration_counter = all_vars[14]
    k1_total_loss = all_vars[15]
    #############################################################################################################

    print_at_master(f" Initialized Optimizer learning rate: {optimizer.param_groups[0]['lr']}")
        
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

                if is_stabilizing:
                    loss = loss * 0.1  # damp gradients
                    stabilizing_steps_remaining -= 1
                    if stabilizing_steps_remaining <= 0:
                        is_stabilizing = False
                
            optimizer.zero_grad()            
            scaler.scale(loss).backward()
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

            ############################################## LoRA check and implementation ########################################
            should_switch = torch.tensor([0], device=f"cuda:{local_rank}")

            if (iteration_counter % args.k1_steps == 0 and not is_dora_initialized) or int(get_total_iteration_count(1.28e6, world_size, args.epochs, args.batch_size) * (1 / 2)) == int(iteration_counter):
                k1_selected_weight_norms, k1_selected_grad_norms = get_selected_modules_norms(model, selected_modules)

                k1_selected_weight_norms = average_module_norms_across_gpus(k1_selected_weight_norms, local_rank)
                k1_selected_grad_norms = average_module_norms_across_gpus(k1_selected_grad_norms, local_rank)

                stored_k1_weight_norms[iteration_counter] = k1_selected_weight_norms
                stored_k1_grad_norms[iteration_counter] = k1_selected_grad_norms
                stored_k1_losses[iteration_counter] = k1_total_loss/args.k1_steps

                k1_total_loss = 0.0

            if global_rank == 0 and not is_dora_initialized and int(get_total_iteration_count(1.28e6, world_size, args.epochs, args.batch_size) * (1 / 2)) == int(iteration_counter):
                should_switch[0] = 1
            
            dist.broadcast(should_switch, src=0)
            
            if should_switch.item() == 1 and not is_dora_initialized:
                stored_k1_losses = check_and_remove_unnecessary_metric(stored_k1_losses, args.consecutive_windows)
                stored_k1_weight_norms = check_and_remove_unnecessary_metric(stored_k1_weight_norms, args.consecutive_windows)
                stored_k1_grad_norms = check_and_remove_unnecessary_metric(stored_k1_grad_norms, args.consecutive_windows)
                
                dora_initialize_start = time.time()
                model, optimizer, scheduler, targeted_lora_parent_modules = instantiate_dora_model(model, optimizer, scheduler, args, selected_modules, stored_k1_grad_norms, iteration_counter)
                dora_initialize_end = time.time()

                is_dora_initialized = True

                if global_rank == 0:
                    model.module.print_trainable_parameters()

            freeze_trigger = torch.tensor([0], device=f"cuda:{local_rank}")

            if global_rank == 0 and is_dora_initialized and not is_frozen and int(get_total_iteration_count(1.28e6, world_size, args.epochs, args.batch_size) * (2 / 3)) == int(iteration_counter):
                freeze_trigger[0] = 1

            dist.broadcast(freeze_trigger, src=0)
            
            if freeze_trigger.item() == 1 and not is_frozen:
                base_model = model.module

                for module in base_model.modules():
                    if all(hasattr(module,attr) for attr in ['lora_A','lora_B', 'base_layer']):
                        for param in module.base_layer.parameters():
                            param.requires_grad = False                

                is_frozen = True

                model = to_ddp(base_model, args, local_rank)

                prev_lr = optimizer.param_groups[0]['lr']
                optimizer = update_optimizer(model, prev_lr, args)
                scheduler = update_scheduler(args, optimizer, scheduler, world_size, iteration_counter)

                is_stabilizing = True
                stabilizing_steps_remaining = 500 

                if global_rank == 0:
                    model.module.print_trainable_parameters()
                
                print_at_master(f" When frozen Optimizer learning rate: {optimizer.param_groups[0]['lr']}")

            #################################################################### LoRA implementation ends #############################################################
            iteration_counter += 1

            del input, target
            torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start_time
        print_at_master(f"Iteration counter: {iteration_counter}")
        print_at_master(f"Total Iteration: {get_total_iteration_count(1.28e6, world_size, args.epochs, args.batch_size)}")

        dist.all_reduce(correct_top1, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_top5, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        train_top1_acc = 100.0 * correct_top1.item() / total_samples.item()
        train_top5_acc = 100.0 * correct_top5.item() / total_samples.item()

        #validation epoch
        if global_rank == 0 and is_dora_initialized:
            print_at_master("before validation")
            model.module.print_trainable_parameters()
        
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
            val_top1_losses.append(top1_loss)
            val_top5_losses.append(top5_loss)
            epoch_losses.append(total_loss/num_batches)

            save_model(args, model, optimizer, scheduler, epoch, scaler, val_top1_losses, val_top5_losses, epoch_losses,is_dora_initialized, is_frozen, 
               targeted_lora_parent_modules, k1_total_loss, iteration_counter, stored_k1_weight_norms,stored_k1_grad_norms, stored_k1_losses)

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

            print_at_master(f"Logits max: {outputs.logits.max().item()}")
            print_at_master(f"Logits min {outputs.logits.min().item()}")
            print_at_master(f"Logits std {outputs.logits.std().item()}")

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