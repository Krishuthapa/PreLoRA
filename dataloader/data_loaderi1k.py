import os

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import Subset

import random

import numpy as np
from PIL import ImageDraw

from timm.data.loader import OrderedDistributedSampler

import torch.distributed as dist

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

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


def setup_distrib(local_rank=0):
    if num_distrib() > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n
    return rt

def num_distrib():
    return int(os.environ.get('WORLD_SIZE', 0))

def create_data_loaders(args):
    data_path_train = os.path.join(args.data_path, 'train')
    
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(args.image_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=args.randaugment_num_ops, magnitude=args.randaugment_magnitude),
    transforms.ToTensor(),
    transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    ])

    data_path_val = os.path.join(args.data_path, 'valid')
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225])
    ])

    set_seed(100)

    train_dataset = ImageFolder(data_path_train, transform=train_transform)
    val_dataset = ImageFolder(data_path_val, transform=val_transform)

    ########################### Subsets ##############################################################
    # train_indices = np.random.choice(len(train_dataset), size=50000, replace=False)
    # val_indices = np.random.choice(len(val_dataset), size=2000, replace=False)

    # train_dataset = Subset(train_dataset, train_indices)
    # val_dataset = Subset(val_dataset, val_indices)
    ########################### Subsets ##############################################################

    print_at_master("length subset train dataset: {}".format(len(train_dataset)))
    print_at_master("length subset val dataset: {}".format(len(val_dataset)))

    rank, world_size = get_dist_info()

    sampler_train = None
    sampler_val = None
    if num_distrib() > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
        sampler_val = OrderedDistributedSampler(val_dataset)

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=sampler_train is None,
        num_workers=args.num_workers, pin_memory=True, sampler=sampler_train, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, sampler=sampler_val)

    # train_loader = PrefetchLoader(train_loader)
    # val_loader = PrefetchLoader(val_loader)
    return train_loader, val_loader


class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        first = True
        for batch in self.loader:
            with torch.cuda.stream(self.stream):  # stream - parallel
                self.next_input = batch[0].cuda(non_blocking=True) # note - (0-1) normalization in .ToTensor()
                self.next_target = batch[1].cuda(non_blocking=True)

            if not first:
                yield input, target  # prev
            else:
                first = False

            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target

            # Ensures that the tensor memory is not reused for another tensor until all current work queued on stream are complete.
            input.record_stream(torch.cuda.current_stream())
            target.record_stream(torch.cuda.current_stream())

        # final batch
        yield input, target

        # cleaning at the end of the epoch
        del self.next_input
        del self.next_target
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    def set_epoch(self, epoch):
        self.loader.sampler.set_epoch(epoch)