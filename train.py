# @copyright CEA-LIST/DIASI/SIALV/LVA (2021)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import os

import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from model.architectures.resnet import construct_resnet
from model.architectures.wide_resnet import wide_resnet
from model.architectures.normalized import NormalizedModel
from utils.utils import load_model, save_model, test, train
from utils.losses import AdversarialDomainAdaptation, SinkhornDistance, AdversarialSinkhornDivergence
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import numpy as np
import torch.nn as nn

from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack, MomentumIterativeAttack

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], help='dataset = [cifar10/cifar100]')
parser.add_argument('--arch', default='resnet', type=str, choices=['resnet', 'wide'], help='architecture to use')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--pretrained', default=None, type=str, help='use another checkpoint as starting point')
parser.add_argument('--gpu', default=None, type=int, help='gpu id for cuda')
parser.add_argument('--sgd', action='store_true', help='optimizer to choose between SGD and Adam')
parser.add_argument('--da', action='store_true', help='use Domain adaptation in loss')
parser.add_argument('--mixed', action='store_true', help='use mixed adversarial training')
parser.add_argument('--sink', action='store_true', help='use sink div in loss')
parser.add_argument('--sink_eps', '-se', default=1, type=float, help='epsilon for sinkhorn div')
parser.add_argument('--weight_decay', '-w', default=5e-4, type=float, help='weight decay for the optimizer')
parser.add_argument('--batch_size', '-bs', default=128, type=int, help='batch size for training')
parser.add_argument('--filename', '-f', default='resnet.ckpt', help='filename of the saved model')
parser.add_argument('--comment', '-c', help='comment for the tensorboard')
parser.add_argument('--nb_epoch', '-n', type=int, help='nb of epoch')
parser.add_argument('--epoch_adv', default=0, type=int, help='epoch to start adversarial training')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--eps_train', default=8., type=float, help='epsilon used during training')
parser.add_argument('--alpha_train', default=2., type=float, help='alpha used during training')
parser.add_argument('--num_iter', default=7, type=int, help='nb iter of PGD during training')
parser.add_argument('--depth', default=20, type=int, help='resnet depth')

args = parser.parse_args()

if args.gpu is not None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

transform_test_adv = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == 'cifar10':
    train_set = data.Subset(torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True),
                            list(range(45000)))
    val_set = data.Subset(torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_test, download=True),
                          list(range(45000, 50000)))
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 10

    mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


elif args.dataset == 'cifar100':
    train_set = data.Subset(torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True), 
                        list(range(45000)))
    val_set = data.Subset(torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_test, download=True),
                        list(range(45000, 50000)))
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 100

    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1).to(device)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=100, shuffle=True, num_workers=args.workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=args.workers, pin_memory=True)

if args.arch == 'wide':
    net = wide_resnet(num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3).to(device)
else:
    net = construct_resnet(depth=args.depth, num_classes=num_classes).to(device)

net = NormalizedModel(model=net, mean=mean, std=std).to(device)

criterion_class = torch.nn.CrossEntropyLoss().to(device)
if args.da:
    criterion_da = AdversarialDomainAdaptation(device=device, intra=False)
elif args.sink:
    criterion_da = AdversarialSinkhornDivergence(args=args, epsilon=args.sink_eps).to(device)
else:
    criterion_da = torch.nn.CrossEntropyLoss().to(device)
        
if args.sgd:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.arch == 'wide':
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,150,230], gamma=0.2)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)

else:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# ###################################### Hyperparameters ######################
nb_epoch = args.nb_epoch
tensorboard_comment = "{}".format(args.comment)
filename = "{}".format(args.filename)

if args.resume and os.path.exists(os.path.join("model", "checkpoints", filename)):
    net, optimizer, best_acc, start_epoch = load_model(net, optimizer, os.path.join("model", "checkpoints"), filename, device)
elif args.pretrained is not None and os.path.exists(os.path.join("model", "checkpoints", args.pretrained)):
    net, optimizer, best_acc, start_epoch = load_model(net, optimizer, os.path.join("model", "checkpoints"), args.pretrained, device)
else:
    start_epoch = 0
    best_acc = 0

# #############################################################################

e = 8.
epsilon = e/255.
max_iter= int(min(e+4, 1.25*e))

def normalize(img, mean=mean, std=std):
    img_n = img - mean
    img_n = img_n / std
    return img_n

adversary = PGDAttack(lambda x: net(x), eps=epsilon, nb_iter=7, ord=np.inf, eps_iter=epsilon/4.)

writer = SummaryWriter(comment=tensorboard_comment)

for epoch in range(start_epoch+1, nb_epoch+1):
    if epoch >= args.epoch_adv:
        train_acc, train_loss = train(epoch, net, train_loader, optimizer, criterion_da, args, adv_training=True, epsilon=args.eps_train/255., alpha=args.alpha_train/255., num_iter=args.num_iter)
    else:
        train_acc, train_loss = train(epoch, net, train_loader, optimizer, criterion_class, args, adv_training=False)
    net.eval()
    val_acc, val_loss = test(net, val_loader, criterion_class, args)

    # adv_acc, adv_loss, _, _ = adv_test(net, val_loader, criterion_class, adversary, epsilon, args, store_imgs=False)
    # writer.add_scalar('adv_acc', adv_acc, epoch)

    writer.add_scalar('train_acc', train_acc, epoch)
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('val_acc', val_acc, epoch)
    writer.add_scalar('val_loss', val_loss, epoch)
    save_model(val_acc, net, optimizer, epoch, os.path.join("model", "checkpoints"), filename)
    if args.sgd:
        scheduler.step()
    
test_acc, test_loss = test(net, test_loader, criterion_class, args)