import os
import torch
import torch.distributed as dist

def get_dist_info():
    initialized = dist.is_available() and dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def is_master():
    rank, _ = get_dist_info()
    return rank == 0


def print_at_master(str):
    if is_master():
        print(str)


def setup_distrib(args):
    if num_distrib() > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)

        
def to_ddp(model,args, local_rank = None):
    if num_distrib() > 1:
        if local_rank is not None:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
            return model

        if local_rank in args:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    return model


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n
    return rt

def num_distrib():
    return int(os.environ.get('WORLD_SIZE', 0))
