# Python
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import os
import torch.optim.lr_scheduler as lr_scheduler
# Custom
import models.resnet as resnet
from models.query_models import TDNet
from models.opt import GDSGD
from train_test.train_test import train, test
from data.load_dataset import load_dataset
from methods.selection_methods import query_samples
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="cifar10",
                    help="cifar10 / cifar100")
parser.add_argument("-i", "--imb_factor", type=int, default=1,
                    help="1 / 10 / 100")
parser.add_argument("-c", "--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t", "--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("--seed", type=int, default=0,
                    help="Training seed.")
parser.add_argument("--subset", type=int, default=5000,
                    help="The size of subset.")
parser.add_argument("-w", "--num_workers", type=str, default=0,
                    help="The number of workers.")
parser.add_argument("--init_dist", type=str, default='random',
                    help="uniform / random.")
args = parser.parse_args()

# Seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# balanced setting
if args.imb_factor == 1:
    args.add_num = {
        'cifar10': 500,
        'cifar100': 1000,
    }[args.dataset]
else:
    args.add_num = {
        'cifar10': 500,
        'cifar100': 1000,
    }[args.dataset]

args.subset = {
    'cifar10': 5000,
    'cifar100': 10000,
}[args.dataset]

args.initial_size = args.add_num

# Main
if __name__ == '__main__':
    datasets = ['cifar10', 'cifar100']
    os.makedirs('../results', exist_ok=True)
    txt_name = f'../results/results1_{args.dataset}_{str(args.imb_factor)}.txt'
    results = open(txt_name, 'w')
    print(txt_name)
    print("Dataset: %s" % args.dataset)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles

    for trial in range(TRIALS):
        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args)
        print('The entire datasize is {}'.format(len(data_train)))
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:args.initial_size]
        unlabeled_set = [x for x in indices if x not in labeled_set]

        # Model - create new instance for every trial so that it resets
        resnet_ = resnet.ResNet18(NUM_CLASS).cuda()
        out_dim = NO_CLASSES
        pred_module = TDNet(out_dim=NUM_CLASS).cuda()
        models = {'backbone': resnet_, 'module': pred_module}
        torch.backends.cudnn.benchmark = True

        for cycle in range(CYCLES):
            train_loader = DataLoader(data_train, batch_size=BATCH,
                                      sampler=SubsetRandomSampler(labeled_set),
                                      pin_memory=True)
            test_loader = DataLoader(data_test, batch_size=BATCH,
                                     pin_memory=True)
            dataloaders = {'train': train_loader, 'test': test_loader}
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = GDSGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = GDSGD(models['module'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}
            # Training and testing
            direc = train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, cycle, EPOCHL)
            acc = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,CYCLES, len(labeled_set), acc))
            np.array([trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")

            if cycle == (CYCLES - 1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:args.subset]
            arg = query_samples(models, data_unlabeled, subset, direc)
            new_list = list(torch.tensor(subset)[arg][-args.add_num:].numpy())
            labeled_set += new_list
            unlabeled_set = [x for x in indices if x not in labeled_set]
            print(len(labeled_set), min(labeled_set), max(labeled_set))

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True, num_workers=args.num_workers)

    results.close()
