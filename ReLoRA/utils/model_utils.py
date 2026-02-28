import os
import torch

from helper_functions.distributed import print_at_master

from transformers import ViTConfig, ViTForImageClassification
from peft import get_peft_model, LoraConfig, TaskType


def get_parsed_args(argparse):
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
    parser.add_argument("--grad_accum_steps", default=2,type=int)
    parser.add_argument("--freeze_warmup_steps", default=300,type=int)
    parser.add_argument("--dora_warmup_steps", default=1000,type=int)


    # Mixup / CutMix
    parser.add_argument('--has_mixup', default = 1, type= int)
    parser.add_argument('--mixup_alpha', default=0.0, type=float)
    parser.add_argument('--cutmix_alpha', default=0.0, type=float)
    parser.add_argument('--mixup_prob', default=1.0, type=float)
    parser.add_argument('--switch_prob', default=0.0, type=float)

    # RandAugment
    parser.add_argument('--randaugment_num_ops', default=2, type=int)
    parser.add_argument('--randaugment_magnitude', default=9, type=int)

    # ReLoRA Arguements
    parser.add_argument('--relora_switch_step', default=15000, type=int)
    parser.add_argument('--relora_rank', default=128, type=int)
    parser.add_argument('--merge_and_reinit_step', default=1500, type=int)

    ## DoRA arguments
    parser.add_argument("--consecutive_windows", default=3,type=int)
    parser.add_argument("--k1_steps", default=1000,type=int)
    parser.add_argument("--k1_loss_thr", default=10.0,type=float)
    parser.add_argument("--k1_wc_thr", default=5.0,type=float)
    parser.add_argument("--lora_scaling", default=2, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    parser.add_argument("--low_rank", default=4, type=int)
    parser.add_argument("--high_rank", default=32, type=int)
    parser.add_argument("--default_rank", default=8, type=int)

    parser.add_argument("--has_lora", default=0, type=int)
    parser.add_argument("--lora_cfg_pth_bc", type=str)
    parser.add_argument("--lora_cfg_pth_ac", type=str)
    parser.add_argument("--use_dora", default=0, type=int)

    parser.add_argument("--checkpoint_dir", default="/lus/grand/projects/datascience/kthapa/vit-lucidrain/ReLoRA", type=str)
    parser.add_argument("--selected_modules", default="attention.query,attention.value", type=str)

    return parser


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

    if bool(args.has_lora) and os.path.exists(args.lora_cfg_pth_ac):
        lora_config_checkpoint = torch.load(args.lora_cfg_pth_ac, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    elif bool(args.has_lora) and os.path.exists(args.lora_cfg_pth_bc):
        lora_config_checkpoint = torch.load(args.lora_cfg_pth_bc, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        lora_config_checkpoint = None


    if bool(args.has_lora) and lora_config_checkpoint:    
        lora_config = lora_config_checkpoint['peft_config']

        print_at_master(f"Lora Conifg:{lora_config}")
            
        peft_config = LoraConfig(
            inference_mode= lora_config['inference_mode'],
            target_modules=lora_config['target_modules'],
            r=lora_config.get('default_r', None),
            rank_pattern = lora_config.get('rank_pattern', {}) or {},
            alpha_pattern = lora_config.get('alpha_pattern', {}) or {},
            lora_alpha = lora_config.get('lora_alpha', None),
            lora_dropout = lora_config.get('lora_dropout', None),
            use_dora=lora_config.get('use_dora', bool(args.use_dora)))
            
        model = get_peft_model(model,peft_config)
        
        if local_rank == 0:
            model.print_trainable_parameters()

    return model


def save_model(args, model, optimizer, scheduler, epoch, iteration_counter, has_convergence_passed, is_model_warmed_up, stored_k1_losses, stored_k1_weight_norms, k1_total_loss, switch_model_start_iter):
    trainable_params = [name for name, param in model.module.named_parameters() if param.requires_grad]
    
    if hasattr(scheduler, 'original_scheduler'):
        scheduler_to_save = scheduler.original_scheduler
    else:
        scheduler_to_save = scheduler

    torch.save({
            'model_state_dict':model.module.state_dict() if model.module is not None else mode.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict': scheduler_to_save.state_dict(),
            'is_model_warmup_done': is_model_warmed_up,
            'trainable_params': trainable_params,
            'epoch':epoch,
            'has_convergence_passed': has_convergence_passed,
            'k1_total_loss':k1_total_loss,
            'iteration_counter': iteration_counter,
            'stored_k1_weight_norms': stored_k1_weight_norms,
            'stored_k1_losses':stored_k1_losses,
            'learning_rate': optimizer.param_groups[0]['lr'] ,
            'switch_model_start_iter': switch_model_start_iter,
    },"{}/vit_checkpoint_{}_{}.pth".format(args.checkpoint_dir, args.model_name,epoch))


def get_peft_config(selected_modules, args):
    peft_config = LoraConfig(
                inference_mode= False,
                target_modules=selected_modules,
                r=args.relora_rank,
                rank_pattern = {},
                alpha_pattern = {},
                lora_alpha = args.lora_scaling,
                lora_dropout = args.lora_dropout,
                use_dora=bool(args.use_dora))
    
    peft_config_store = {
                'inference_mode': False,
                'target_modules': selected_modules,
                'default_r': args.relora_rank,
                'rank_pattern': None,
                'alpha_pattern': None,
                'lora_alpha': args.lora_scaling,
                'lora_dropout': args.lora_dropout,
                'use_dora':bool(args.use_dora)
                }
                
    return peft_config, peft_config_store