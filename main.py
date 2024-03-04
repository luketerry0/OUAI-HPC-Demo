import os
import torch
import torchvision
from torchvision.transforms import v2
from torchvision import transforms
import time
from copy import deepcopy as copy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import (enable_wrap,
                                         size_based_auto_wrap_policy, wrap)
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from tqdm import tqdm
import time 
import gc
import functools
from copy import deepcopy as copy
import time
import wandb
import argparse

def dino_model():
    os.environ['TORCH_HOME'] = './'
    # DINOv2 vit-s (14) with registers
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    # state = model.state_dict()
    # mymodel = vit_small(14, 4)
    # mymodel.load_state_dict(state)
    model.eval()

    return model.to('cpu')

def dino_transforms():
    return v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    )


def train(model, optimizer, loader):
    model.train()
    loss = torch.nn.CrossEntropyLoss()

    for i, (X, y) in tqdm(enumerate(loader)):
        out = model(X.to(int(os.environ['RANK']) % torch.cuda.device_count()))
        optimizer.zero_grad()
        l = loss(out, y.to(int(os.environ['RANK']) % torch.cuda.device_count()))
        l.backward()
        optimizer.step()
        

def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def epoch_accuracy(loader_s, student):
    student.eval()

    out_epoch_s = [accuracy(student(L.to(int(os.environ['RANK']) % torch.cuda.device_count())), y)[0].detach().cpu().item() for L, y in loader_s]

    student.train()

    return sum(out_epoch_s) / len(out_epoch_s)

def test(network, test_loader, dtype=torch.float32, silent=False):
    network.eval()
    test_loss = 0
    correct = 0
    test_losses=[]

    device = int(os.environ['RANK']) % torch.cuda.device_count()
    total = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(device))
            test_loss += torch.nn.CrossEntropyLoss()(output.to(device), target.to(device)).item()
            pred = output.data.max(1, keepdim=True)[1].cpu()
            pred, target = pred.cpu(), target.cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
            total += torch.ones_like(target).sum()
        test_loss /= total
        test_losses.append(test_loss)
        if not silent:
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / total))
            
    if silent:
        return 100. * correct / total


def latency(f, x, trials = 100):
    f.cpu()
    total = 0.0
    for trial in range(trials):
        start = time.perf_counter()
        f(x)
        total += time.perf_counter() - start
    return total / trials


class LinearProbe(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.m = module
        self.linear = torch.nn.Linear(1024, 100)
    
    def forward(self, x):
        x = self.m(x).detach()
        return self.linear(x)


class FPFT(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.m = module.m
        self.linear = module.linear
    
    def forward(self, x):
        x = self.m(x)
        return self.linear(x) / 8
    

def setup():
    dist.init_process_group("nccl", rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))

def cleanup():
    dist.destroy_process_group()


def training_process(args):

    rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])

    device = torch.device(rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    if rank == 0:
       wandb.init(project='OUAI Demo',
                   entity='ai2es',
                   name=f'Tuning DINOv2',
        )
       
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=10_000_000)

    DINOv2 = dino_model()

    model = LinearProbe(DINOv2).to(device)

    model = FSDP(model, 
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True)
            )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    DINOv2_transform = dino_transforms()

    train_ds = torchvision.datasets.CIFAR100('./cifar100', train=True, transform=DINOv2_transform, download=True)
    val_ds = torchvision.datasets.CIFAR100('./cifar100', train=False, transform=DINOv2_transform, download=True)

    sampler_train = DistributedSampler(train_ds, rank=rank, num_replicas=world_size, shuffle=True)
    sampler_val = DistributedSampler(val_ds, rank=rank, num_replicas=world_size, shuffle=False)

    cuda_kwargs = {'num_workers': 12, 
                    'pin_memory': True, 
                    'shuffle': False}

    loader_kwargs = {'batch_size': args.batch_size, 
                     'sampler': sampler_train, 
                     'multiprocessing_context': 'forkserver', 
                     'persistent_workers': False,
                     'drop_last': False}
    
    loader_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_ds, **loader_kwargs)

    loader_kwargs = {'batch_size': args.batch_size, 
                     'sampler': sampler_val, 
                     'multiprocessing_context': 'forkserver', 
                     'persistent_workers': False,
                     'drop_last': False}
    
    loader_kwargs.update(cuda_kwargs)

    val_loader = DataLoader(val_ds, **loader_kwargs)

    s = 0
    print(os.environ['RANK'], 'linear probe')
    for epoch in range(args.epochs):
        print(os.environ['RANK'], epoch)
        s += 1
        train(model, opt, train_loader)
        gc.collect()
        vacc = torch.tensor(test(model, val_loader, silent=True)).to(device)
        dist.all_reduce(vacc, op=dist.ReduceOp.AVG)
        if rank == 0:
            wandb.log({'validation accuracy': vacc})
            print({'validation accuracy': vacc})


    states = model.state_dict()

    DINOv2 = dino_model()

    model = LinearProbe(DINOv2).to(device)

    model.load_state_dict(states)

    model = FPFT(model)

    model = FSDP(model, 
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True)
            )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr / 8.0)

    print(os.environ['RANK'], 'finetuning')
    for epoch in range(args.epochs):
        s += 1
        train(model, opt, train_loader)
        gc.collect()
        vacc = torch.tensor(test(model, val_loader, silent=True)).to(device)
        dist.all_reduce(vacc, op=dist.ReduceOp.AVG)
        if rank == 0:
            wandb.log({'validation accuracy': vacc})
            print({'validation accuracy': vacc})


def create_parser():
    parser = argparse.ArgumentParser(description='MCT benchmark')
    
    parser.add_argument('-e', '--epochs', type=int, default=5, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=256, 
                        help='batch size for training (default: 64)')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                        help='learning rate for SGD (default 1e-3)')

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    setup()
    training_process(args)
    cleanup()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method('spawn')

    main()