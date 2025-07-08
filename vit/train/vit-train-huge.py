
import sys
import os

import math

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

from dataloader.data_loaderi21k import create_data_loaders
from helper_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib, get_dist_info
from helper_functions.general_helper_functions import accuracy, AverageMeter, silence_PIL_warnings

from helper_functions.optimizers import create_optimizer_adam
from helper_functions.losses import CrossEntropyLS, SoftTargetCrossEntropy

from transformers import ViTConfig, ViTForImageClassification

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
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--wandb_name", default='vit-huge', type=str)
parser.add_argument("--model_name", default='HUGE', type=str)


# Mixup / CutMix
parser.add_argument('--has_mixup', default = 1, type= int)
parser.add_argument('--mixup_alpha', default=0.0, type=float)
parser.add_argument('--cutmix_alpha', default=0.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)
parser.add_argument('--switch_prob', default=0.0, type=float)

# RandAugment
parser.add_argument('--randaugment_num_ops', default=2, type=int)
parser.add_argument('--randaugment_magnitude', default=9, type=int)


def get_mixup(args):
    mixup_fn = Mixup(
    mixup_alpha=args.mixup_alpha,   # Mixup strength
    cutmix_alpha=args.cutmix_alpha,  # Disable CutMix
    cutmix_minmax=None,
    prob=args.mixup_prob,          # Always apply Mixup
    switch_prob=args.switch_prob,   # No switching to CutMix since disabled
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

    print_at_master("model : {} , patch: {} and img_size: {}".format(args.model_name, args.patch_size, args.img_size))
    print_at_master("nlayers: {}, dim : {} , heads: {} and mlp_dim: {}".format(args.nlayers, args.hidden_dim, args.nheads, args.ffn_dim))

    set_seed(100)

    model_config = ViTConfig(hidden_size = args.hidden_dim, 
                    intermediate_size = args.ffn_dim,
                    image_size = args.img_size,
                    patch_size = args.patch_size,
                    num_attention_heads = args.nheads,
                    num_hidden_layers = args.nlayers,
                    num_labels = args.num_classes)

    model = ViTForImageClassification(model_config).to(local_rank)
    model = to_ddp(model,args,local_rank)

    if global_rank == 0:
        wandb.login()
        wandb.init(project=args.wandb_name, config = args)
        wandb.watch(model)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.999))
    train_loader, val_loader = create_data_loaders(args)

    model.train()
    train_21k(model, train_loader, val_loader, optimizer, args)

def get_scheduler(args, optimizer, world_size):
    steps_per_epoch = int(1.235e7 // (args.batch_size * world_size))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

def verify_model_sync(model, rank, world_size, eps=1e-6):
    with torch.no_grad():
        local_sum = torch.tensor([sum(p.sum().item() for p in model.parameters())], device='cuda')
        global_sum = local_sum.clone()
        torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)

        average = global_sum.item() / world_size
        if abs(local_sum.item() - average) > eps:
            print_at_master(f"[Rank {rank}] ❌ Model desynchronized: Local sum = {local_sum.item()}, Avg = {average}")
            return False
        if rank == 0:
            print_at_master(f"[Rank {rank}] ✅ Model synchronized after check.")
        return True

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

    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        val_top1_losses = checkpoint['val_top1_losses'].tolist()
        val_top5_losses = checkpoint['val_top5_losses'].tolist()
        epoch_losses = checkpoint['epoch_losses'].tolist()
    else:
        start_epoch = 0
        val_top1_losses = []
        val_top5_losses = []
        epoch_losses = []

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
                
            optimizer.zero_grad()            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # if i% 100 == 0 and i > 0 and global_rank == 0:
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             print_at_master(f"[{name}] grad mean: {param.grad.abs().mean().item()}")
            #     print_at_master("=====================================================================")
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            loss_collector = loss.detach().clone()

            if num_distrib() > 1:
                dist.all_reduce(loss_collector, op=dist.ReduceOp.SUM)
            
            avg_loss_acc_gpus = loss_collector / world_size

            total_loss += avg_loss_acc_gpus.item()
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
            
            del input, target
            torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start_time

        dist.all_reduce(correct_top1, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_top5, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        train_top1_acc = 100.0 * correct_top1.item() / total_samples.item()
        train_top5_acc = 100.0 * correct_top5.item() / total_samples.item()

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

        if ((epoch> 0 and epoch% 1 == 0) or epoch == (args.epochs-1)) and global_rank == 0:
            val_top1_losses.append(top1_loss)
            val_top5_losses.append(top5_loss)
            epoch_losses.append(total_loss/num_batches)

            torch.save({
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch':epoch,
                    'scaler':scaler.state_dict(),
                    'val_top1_losses': torch.tensor(val_top1_losses),
                    'val_top5_losses': torch.tensor(val_top5_losses),
                    'epoch_losses': torch.tensor(epoch_losses)
            },"/lus/grand/projects/datascience/kthapa/vit-lucidrain/checkpoint_huge/vit_checkpoint_{}_{}.pth".format(args.model_name,epoch))

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
        print("Total 1 avg", top1_sum)
        print("Total 5 avg", top5_sum)
        print("Total exampled", total)
        print(f"Validation Top-1 Accuracy: {top1_avg:.2f}%")
        print(f"Validation Top-5 Accuracy: {top5_avg:.2f}%")

    return top1_avg, top5_avg

if __name__ == '__main__':
    main()
