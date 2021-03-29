# @copyright CEA-LIST/DIASI/SIALV/LVA (2021)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import argparse
from tqdm import tqdm
import os
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from utils.utils import load_model, adv_test, test
import numpy as np
from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack, MomentumIterativeAttack, GradientSignAttack, FastFeatureAttack
from fast_adv.attacks import DDN, CarliniWagnerL2
from model.architectures.resnet import construct_resnet
from model.architectures.wide_resnet import wide_resnet
from model.architectures.normalized import NormalizedModel
from model.architectures.resnet_model_pcl import construct_resnet_pcl

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Adversarial Test')
parser.add_argument('--gpu', default=1, type=int, help='gpu id for cuda')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--arch', default='resnet', type=str, choices=['resnet', 'wide'], help='architecture to use')
parser.add_argument('--depth', default=20, type=int, help='resnet depth')
parser.add_argument('--eps', default=4, type=float, help='epsilon of the attack')
parser.add_argument('--attack', '-a', default='PGD', type=str, help='Attack used', choices=['PGD', 'MIFGSM', 'FGSM', 'CW', 'WPGD', 'DDN'])
parser.add_argument('--filename', '-f', default='resnet.ckpt', type=str, help='Model to load')
parser.add_argument('--norm', '-l', default=8, type=int, help='norm for adv (2=L2 / 8=Linf)', choices=[8,2])
parser.add_argument('--store_imgs', action='store_true', help='store first batch of imgs')
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

if args.norm == 8:
    norm = np.inf
else:
    norm = 2

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10

    mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


elif args.dataset == 'cifar100':
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1).to(device)


test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

# instantiate the model
if args.arch == 'wide':
    net = wide_resnet(num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3).to(device)
elif args.arch == 'pcl':
    net = construct_resnet_pcl(num_classes=num_classes, depth=110)
else:
    net = construct_resnet(depth=args.depth, num_classes=num_classes).to(device)

filename = args.filename
# print(best_acc, epoch)
pcl = False

if filename[-1] == 'h':
    mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1).to(device)
    net = NormalizedModel(model=net, mean=mean, std=std).to(device)
    net.load_state_dict(torch.load(os.path.join("model", "checkpoints", filename)))
elif args.arch == 'pcl':
    net = torch.nn.DataParallel(net).cuda()
    pcl = True
    checkpoint = torch.load(os.path.join("model", "checkpoints", filename))
    net.load_state_dict(checkpoint['state_dict'])
else:
    net = NormalizedModel(model=net, mean=mean, std=std).to(device)
    net, optim, best_acc, epoch = load_model(net, None, os.path.join("model", "checkpoints"), filename, device)

net = net.eval()

def wrapper(x, pcl=False):
    if pcl:
        return net(normalize(x))[3]
    else:
        return net(x)

def normalize(img, mean=mean, std=std):
    img_n = img - mean
    img_n = img_n / std
    return img_n

def unnormalize(img, mean=mean, std=std):
    img_u = img*std
    img_u = img_u + mean
    return img_u

epsilon = args.eps
epsilon = epsilon/255.
ddn = False
if args.attack == 'PGD':
    adversary = PGDAttack(lambda x: wrapper(x, pcl=pcl), eps=epsilon, eps_iter=epsilon/4, nb_iter=10, ord=norm, rand_init=True)
elif args.attack == 'MIFGSM':
    adversary = MomentumIterativeAttack(lambda x: wrapper(normalize(x), pcl=pcl), eps=epsilon, eps_iter=epsilon/10, ord=norm, nb_iter=10)
elif args.attack == 'FGSM':
    adversary = GradientSignAttack(lambda x: wrapper(x, pcl=pcl), eps=epsilon)
    # adversary = PGDAttack(lambda x: wrapper(x, pcl=pcl), eps=epsilon, eps_iter=epsilon, nb_iter=1, ord=norm, rand_init=False)
elif args.attack == 'CW':
    adversary = CarliniWagnerL2Attack(lambda x: wrapper(x, pcl=pcl), 10, binary_search_steps=2, max_iterations=500, initial_const=1e-1)
elif args.attack == 'DDN':
    adversary = DDN(steps=100, device=device)
    ddn = True
else:
    adversary = None

criterion = torch.nn.CrossEntropyLoss()
net.eval()

test_acc_adv, test_loss_adv, dist_l2, dist_linf = adv_test(lambda x: wrapper(x, pcl=pcl), test_loader, criterion, adversary, epsilon, args, ddn=ddn, store_imgs=args.store_imgs)
test_acc, test_loss = test(lambda x: wrapper(x, pcl=pcl), test_loader, criterion, args)

print(f"Original Accuracy : {test_acc}")
print(f"Accuracy under {args.attack} with eps={args.eps} : {test_acc_adv}")
print("=== L2 ===")
print(f"Median distance : {torch.median(dist_l2)} | Mean distance : {torch.mean(dist_l2)} | Max distance : {torch.max(dist_l2)}")
print("=== Linf ===")
print(f"Median distance : {torch.median(dist_linf)} | Mean distance : {torch.mean(dist_linf)} | Max distance : {torch.max(dist_linf)}")