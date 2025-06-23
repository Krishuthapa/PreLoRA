
import sys
import os

import torch
import torch.optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data.distributed

from torch.amp import GradScaler, autocast

import time
import argparse

from transformers import ViTConfig, ViTForImageClassification

from dataloader.data_loader2 import create_data_loaders
from helper_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib, get_dist_info
from helper_functions.general_helper_functions import accuracy, AverageMeter, silence_PIL_warnings, HookController, check_container_and_assign

from helper_functions.optimizers import create_optimizer_adam
from helper_functions.losses import CrossEntropyLS
from helper_functions.convergence import check_partial_convergence, check_and_remove_unnecessary_metric, check_activation_change
#from helper_functions.dora_instantiate import instantiate_dora_model

from utils.param_utils import get_selected_modules_weights

import wandb

parser = argparse.ArgumentParser(description='PyTorch ImageNet21K Single-label Training From Random Initialization')
parser.add_argument('--data_path', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--num_classes', default=11221, type=int)
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
parser.add_argument("--model_name", default='LARGE', type=str)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--wandb_name", default='vit-hf', type=str)

#### DORA arguments
parser.add_argument("--consecutive_windows", default=3,type=int)
parser.add_argument("--k1_steps", default=1000,type=int)
parser.add_argument("--k1_loss_thr", default=10.0,type=float)
parser.add_argument("--k1_wc_thr", default=5.0,type=float)
parser.add_argument("--k2_steps", default=100,type=int)
parser.add_argument("--k2_freeze_thr", default=10, type=float)
parser.add_argument("--lora_scaling", default=2, type=int)
parser.add_argument("--low_rank", default=4, type=int)
parser.add_argument("--high_rank", default=32, type=int)
parser.add_argument("--default_rank", default=8, type=int)
parser.add_argument("--selected_modules", default="attention.query,attention.value", type=str)

possbile_modules = ['attention.query','attention.key','attention.value', 'intermediate.dense','output.dense', 'attetnion.output.dense']

def main():
    # arguments
    args = parser.parse_args()

    silence_PIL_warnings()

    setup_distrib(args)

    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))

    print_at_master("model : {} , patch: {} and img_size: {}".format(args.model_name, args.patch_size, args.img_size))
    print_at_master("nlayers: {}, dim : {} , heads: {} and mlp_dim: {}".format(args.nlayers, args.hidden_dim, args.nheads, args.ffn_dim))


    model_config = ViTConfig(hidden_size = args.hidden_dim, 
                    intermediate_size = args.ffn_dim,
                    image_size = args.img_size,
                    patch_size = args.patch_size,
                    num_attention_heads = args.nheads,
                    num_hidden_layers = args.nlayers,
                    num_labels = args.num_classes,
                    hidden_dropout_prob = args.dropout,
                    attention_probs_dropout_prob = args.dropout)

    model = ViTForImageClassification(model_config).cuda()
    model = to_ddp(model,args,local_rank)

    if global_rank == 0:
        wandb.login()
        wandb.init(project="vit-hf-dora", config = args)
        wandb.watch(model)

    optimizer = create_optimizer_adam(model, args)
    train_loader, val_loader = create_data_loaders(args)

    train_21k(model, train_loader, val_loader, optimizer, args)

def get_scheduler(args, optimizer, world_size):
    steps_per_epoch = int(1.28e6 // (args.batch_size * world_size))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return scheduler

def train_21k(model, train_loader, val_loader, optimizer, args):

    world_size = int(os.environ.get('WORLD_SIZE'))
    global_rank = int(os.environ.get('RANK'))

    # set loss
    loss_fn = CrossEntropyLS(args.label_smooth)

    # set scheduler
    scheduler = get_scheduler(args, optimizer, world_size)
    
    # set scalaer
    scaler = GradScaler(device='cuda')

    has_hooks = False

    selected_modules = args.selected_modules.split(',')

    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        val_top1_losses = checkpoint['val_top1_losses'].tolist()
        val_top5_losses = checkpoint['val_top5_losses'].tolist()
        epoch_losses = checkpoint['epoch_losses'].tolist()

        stored_k1_weights = check_container_and_assign(checkpoint, 'stored_k1_weights', {})
        stored_k1_losses = check_container_and_assign(checkpoint, 'stored_k1_losses', {})
        is_dora_initialized = check_container_and_assign(checkpoint,'is_dora_initialzied', False)
        targeted_lora_parent_modules = check_container_and_assign(checkpoint,'targeted_lora_parent_modules',[])
        iteration_counter = check_container_and_assign(checkpoint,'iteration_counter',0)
        k1_total_loss = check_container_and_assign(checkpoint, 'k1_total_loss', 0.0)
        
    else:
        start_epoch = 0
        val_top1_losses = []
        val_top5_losses = []
        epoch_losses = []


        stored_k1_weights = {}
        stored_k1_losses = {}
        targeted_lora_parent_modules = []
        is_dora_initialized = False
        iteration_counter = 0
        k1_total_loss = 0.0

    # training loop
    for epoch in range(start_epoch, args.epochs):
        if num_distrib() > 1:
            train_loader.sampler.set_epoch(epoch)
       
        epoch_start_time = time.time()
        
        epoch_total_loss = 0.0
        epoch_num_batches = 0

        for i, (input, target) in enumerate(train_loader):
            with autocast(device_type='cuda'):  # mixed precision
                output = model(input)
                loss = loss_fn(output.logits, target)  # note - loss also in fp16
                
            loss_collector = loss.detach().clone()
            dist.all_reduce(loss_collector, op=dist.ReduceOp.SUM)
            avg_loss_acc_gpus = loss_collector / world_size

            epoch_total_loss += avg_loss_acc_gpus.item()
            k1_total_loss += avg_loss_acc_gpus.item()

            epoch_num_batches += 1

            model.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if i % 1000 == 0:
                print_at_master(f"\n Epoch:{epoch}, index: {i} and loss: {loss.item()} (CE)")

            iteration_counter+=1

            ############################################## LoRA check and implementation ########################################
            
            if iteration_counter % args.k1_steps == 0 and not is_dora_initialized and global_rank == 0:
                k1_selected_weights = get_selected_modules_weights(model, selected_modules)
                
                if iteration_counter not in stored_k1_weights:
                    stored_k1_weights[iteration_counter] = k1_selected_weights
                
                if iteration_counter not in stored_k1_losses:
                    stored_k1_losses[iteration_counter] = k1_total_loss/args.k1_steps
                    k1_total_loss = 0.0
                
                # Always maintains storing given number of consecutive windows values.
                stored_k1_losses = check_and_remove_unnecessary_metric(stored_k1_losses, args.consecutive_windows)
                stored_k1_weights = check_and_remove_unnecessary_metric(stored_k1_weights, args.consecutive_windows)

                is_partially_converged = check_partial_convergence(stored_k1_losses, stored_k1_weights,
                                            thr_loss = args.k1_loss_thr, thr_weight = args.k1_wc_thr, checking_step = args.k1_steps, checking_window = args.consecutive_windows)


                if is_partially_converged:
                        model, targeted_lora_parent_modules = instantiate_dora_model(model, args, selected_modules, stored_k1_weights)
                        is_dora_initialized = True

                        torch.save({
                            'model_state_dict':model.module.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'epoch':epoch,
                            'scaler':scaler.state_dict(),
                            'val_top1_losses': torch.tensor(val_top1_losses),
                            'val_top5_losses': torch.tensor(val_top5_losses),
                            'epoch_losses': torch.tensor(epoch_losses),
                            'is_dora_initialized': is_dora_initialized,
                            'targeted_lora_parent_modules':targeted_lora_parent_modules,
                            'k1_total_loss':k1_total_loss,
                            'iteration_counter': iteration_counter
                        },"/lus/grand/projects/datascience/kthapa/vit-lucidrain/checkpoint_dora/vit_checkpoint_dora_hf_{}_{}.pth".format(args.model_name,epoch))

            if is_dora_initialized and not has_hooks and global_rank == 0:
                for module_name in targeted_lora_parent_modules:
                    hook_ctrl = HookController(interval = args.k2_steps)
                    module = dict(model.named_modules())[module_name]
                    module.register_forward_hook(check_activation_change(name), args.k2_freeze_thr, hook_ctrl)
                
                has_hooks = True

                iteration_counter = 0
                
            if is_dora_initialized and iteration_counter >0 and (iteration_counter % args.k2_steps == 0) and global_rank == 0:
                model.print_trainable_parameters()

            #################################################################### LoRA implementation ends #############################################################

            del input, target
            torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start_time
        
        # validation epoch
        top1_loss, top5_loss = validate_21k(val_loader, model)

        if global_rank == 0:
            wandb.log({'epoch': epoch, 
                'loss':epoch_total_loss/epoch_num_batches, 
                'lr': optimizer.param_groups[0]['lr'],
                'top1_val_acc': top1_loss,
                'top5_val_acc':top5_loss,
                'training_rate [img/sec]': len(train_loader) * args.batch_size / epoch_time * max(num_distrib(),1),
                'epoch_time': epoch_time})

        if (epoch> 0 and epoch% 2 == 0) or epoch == (args.epochs-1):
            val_top1_losses.append(top1_loss)
            val_top5_losses.append(top5_loss)
            epoch_losses.append(epoch_total_loss/epoch_num_batches)
            
            file_path = "/lus/grand/projects/datascience/kthapa/vit-lucidrain/checkpoint_dora/vit_checkpoint_hf_{}_{}.pth".format(args.model_name,epoch)

            torch.save({
                    'model_state_dict':model.module.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch':epoch,
                    'scaler':scaler.state_dict(),
                    'val_top1_losses': torch.tensor(val_top1_losses),
                    'val_top5_losses': torch.tensor(val_top5_losses),
                    'epoch_losses': torch.tensor(epoch_losses),
                    'is_dora_initialized': is_dora_initialized,
                    'targeted_lora_parent_modules':targeted_lora_parent_modules,
                    'k1_total_loss':k1_total_loss,
                    'iteration_counter': iteration_counter
            },file_path)

        model.train()

def validate_21k(val_loader, model):
    print_at_master("starting validation")
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            # mixed precision
            with autocast(device_type='cuda'):
                outputs = model(input)
                logits = outputs.logits.float()
            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            if num_distrib() > 1:
                acc1 = reduce_tensor(acc1, num_distrib())
                acc5 = reduce_tensor(acc5, num_distrib())
                torch.cuda.synchronize()
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

    return top1.avg, top5.avg

if __name__ == '__main__':
    main()