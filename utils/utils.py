# @copyright CEA-LIST/DIASI/SIALV/LVA (2021)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import numpy as np
import torch.nn as nn
from trattack.attack_methods import select_index, tr_attack_adaptive
from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack
from fast_adv.attacks import DDN
from torch.autograd import Variable

import torch.nn.functional as F
from torch.optim import Adam

def load_model(net, optim, save_path, filename, device):
    """Load a model and its optimizer

    Args:
        net (nn.Module): architecture of the saved model
        optim (torch.optim): optimizer to load
        save_path (str): path where the file is stored
        filename (str): filename to open
        device (torch.device): device to load the model and optimizer to

    Returns:
        net (nn.Module): loaded model
        optim (torch.optim): optimizer of the loaded model
        best_acc (int): performance of the model
        epoch (int): number of epoch the model were trained
    """
    state = torch.load(os.path.join(save_path, filename), map_location=device)
    net.load_state_dict(state['net'])
    best_acc = state['acc']
    epoch = state['epoch']
    try :
        optim_state = state['optim']
    except KeyError:
        optim_state = None

    if optim_state and optim:
        optim.load_state_dict(optim_state)
    
    return net, optim, best_acc, epoch

def save_model_best_acc(acc, best_acc, net, optim, epoch, save_path, filename):
    """Save a model and its optimizer if its accuracy is better than the saved one

    Args:
        acc (int): performance of the model to save
        best_acc (int): saved model best performance
        net (nn.Module): model to save
        optim (torch.optim): optimizer of the model to save
        epoch (int): number of epoch the model were trained
        save_path (str): path on disk where to save the model to
        filename (str): filename on disk

    Returns:
        best_acc (int): the saved model best performance
    """
    if acc > best_acc:
        print('Saving ...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optim': optim.state_dict()
        }
        torch.save(state, os.path.join(save_path, filename))
        best_acc = acc
    return best_acc

def save_model(acc, net, optim, epoch, save_path, filename):
    """Save a model and its optimizer

    Args:
        acc (int): performance of the model to save
        net (nn.Module): model to save
        optim (torch.optim): optimizer of the model to save
        epoch (int): number of epoch the model were trained
        save_path (str): path on disk where to save the model to
        filename (str): filename on disk
    """
    print('Saving ...')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'optim': optim.state_dict()
    }
    torch.save(state, os.path.join(save_path, filename))

def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """Construct an adversarial perturbation for a given batch of images X using ground truth y and a given model with a L_inf-PGD attack.

    Args:
        model (nn.Module): model to attack
        X (Tensor): batch of images
        y (Tensor): ground truth
        epsilon (float, optional): perturbation size. Defaults to 0.1.
        alpha (float, optional): step size. Defaults to 0.01.
        num_iter (int, optional): number of iterations. Defaults to 20.
        randomize (bool, optional): random start for the perturbation. Defaults to False.

    Returns:
        Tensor: perturbation to apply to the batch of images
    """
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def train(epoch, net, trainloader, optimizer, criterion, args, adv_training=False, epsilon=8./255, alpha=2./255, num_iter=7):
    """Train a model for an epoch on a given dataset

    Args:
        epoch (int): current epoch of training
        net (nn.Module): model to train
        trainloader (dataloader): dataloader of the training set
        optimizer (torch.optim): training optimizer
        criterion (): training loss
        args ([type]): parser args
        adv_training (bool, optional): perform adversarial training or not. Defaults to False.
        epsilon (float, optional): perturbation size of the attack. Defaults to 8./255.
        alpha (float, optional): step size of the attack. Defaults to 2./255.
        num_iter (int, optional): number of iteration of the attack. Defaults to 7.

    Returns:
        float: training accuracy for this epoch
        float: mean loss on each batch of this epoch
    """
    print(f'\nEpoch: {epoch}')
    train_loss = 0
    correct = 0
    total = 0
    net.train()
    pbar = tqdm(desc="Training", total=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        if adv_training:
            delta = pgd_linf(net, inputs, targets, epsilon=epsilon, alpha=alpha, num_iter=num_iter, randomize=True)
            inputs_adv = inputs + delta
            if args.gpu is not None:
                inputs_adv = inputs_adv.cuda(args.gpu, non_blocking=True)
        net.train()
        optimizer.zero_grad()
        if not adv_training:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        elif args.da:
            outputs_adv = net(inputs_adv)
            outputs = net(inputs)
            loss_clean, loss_adv, loss_mmd, loss_coral = criterion(outputs, outputs_adv, targets)
            loss = loss_clean + loss_adv + (1/2)*(loss_mmd + loss_coral)
        elif args.sink:
            outputs_adv = net(inputs_adv)
            outputs = net(inputs)
            loss_adv, loss_sink = criterion(outputs, outputs_adv, targets)
            loss = loss_adv + loss_sink
        elif args.mixed:
            outputs_adv = net(inputs_adv)
            outputs = net(inputs)
            loss_clean = criterion(outputs, targets)
            loss_adv = criterion(outputs_adv, targets)
            loss = loss_clean + loss_adv
        else:
            outputs_adv = net(inputs_adv)
            loss = criterion(outputs_adv, targets)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        mean_loss = train_loss/(batch_idx+1)
        if not adv_training:
            _, predicted = outputs.max(1)
        else:
            _, predicted = outputs_adv.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc = 100.*correct/total
        pbar.update()
        postfix = {'Loss': mean_loss, 'Acc': train_acc, 'correct': f'{correct}/{total}'}
        if adv_training:
            if args.da:
                postfix['mmd'] = loss_mmd.item()
                postfix['coral'] = loss_coral.item()
            if args.sink:
                postfix['sink'] = loss_sink.item()
        pbar.set_postfix(postfix)
    pbar.close()
    return train_acc, mean_loss

def test(net, testloader, criterion, args):
    # net.eval()
    test_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(desc="Test", total=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            mean_loss = test_loss/(batch_idx+1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            pbar.update()
            pbar.set_postfix({'Loss': mean_loss, 'Acc': acc, 'correct': f'{correct}/{total}'})
        pbar.close()
    return acc, mean_loss

def adv_test(net, testloader, criterion, attack, epsilon, args, ddn=False, store_imgs=False):
    test_loss = 0
    correct = 0
    total = 0
    dist_l2 = []
    dist_linf = []
    pbar = tqdm(desc="Adv Test", total=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # attack.clip_max = inputs.max()
        # attack.clip_min = inputs.min()
        if ddn:
            inputs_adv = attack.attack(net, inputs, targets, targeted=False).cuda(args.gpu, non_blocking=True)
        else:
            attack.clip_max = 1.0
            attack.clip_min = 0.0
            inputs_adv = attack.perturb(inputs, targets).cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            outputs_adv = net(inputs_adv)
        if batch_idx == 0 and store_imgs:
            all_imgs = torch.cat((inputs, inputs_adv))
            save_image(all_imgs, f'adv_examples/{args.filename}_{args.attack}_{args.norm}_{epsilon}.png', nrow=inputs.size(0), pad_value=0)
        tmp_dist_l2 = []
        tmp_dist_linf = []
        for i in range(inputs.size(0)):
            tmp_dist_l2.append(torch.norm(inputs_adv[i,:] - inputs[i,:], p=2))
            tmp_dist_linf.append(torch.max(torch.abs(inputs_adv[i,:] - inputs[i,:])))
        tmp_dist_l2 = torch.stack(tmp_dist_l2)
        tmp_dist_linf = torch.stack(tmp_dist_linf)
        dist_l2.append(tmp_dist_l2)
        dist_linf.append(tmp_dist_linf)
        loss = criterion(outputs_adv, targets)
        test_loss += loss.item()
        mean_loss = test_loss/(batch_idx+1)
        _, predicted = outputs_adv.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        pbar.update()
        pbar.set_postfix({'Loss': mean_loss, 'Acc': acc, 'correct': f'{correct}/{total}'})
    pbar.close()
    dist_l2 = torch.cat(dist_l2)
    dist_linf = torch.cat(dist_linf)
    return acc, mean_loss, dist_l2, dist_linf

def tr_test(net, testloader, eps, c, p, iter, worst_case, batchsize, adap, criterion, device):
    num_d = 10000
    X_ori = torch.Tensor(num_d, 3, 32, 32)
    X_tr_first = torch.Tensor(num_d, 3, 32, 32)
    iter_tr_first = 0.
    Y_test = torch.LongTensor(num_d)

    for i, (data, target) in tqdm(enumerate(testloader), total=len(testloader), desc= "Creating adv examples"):
        X_ori [i * batchsize:(i + 1) * batchsize, :] = data
        Y_test[i * batchsize:(i + 1) * batchsize] = target

        if not adap:
            X_tr_first[i * batchsize:(i + 1) * batchsize,:], a = trattack.tr_attack_iter(net, data, target, args.eps, c = c,
                    p = p, iter = iter, worst_case = worst_case)
            iter_tr_first += a
        else:
            X_tr_first[i * batchsize:(i + 1) * batchsize, :], a = trattack.tr_attack_adaptive_iter(net, data, target, eps, c = c,
                    p = p, iter = iter, worst_case = worst_case)
            iter_tr_first += a

    num_data = X_tr_first.size()[0]
    num_iter = num_data // batchsize
    net.eval()
    correct = 0
    test_loss = 0
    mean_loss = 0
    total = 0
    pbar = tqdm(desc="Adv Test", total=num_data)
    with torch.no_grad():
        for i in range(num_iter):
            data, target = X_tr_first[batchsize * i:batchsize * (i + 1), :], Y_test[batchsize * i:batchsize * (i + 1)]
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            mean_loss = test_loss/(i+1)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            acc = 100.*correct/total
            pbar.update(batchsize)
            pbar.set_postfix({'Loss': mean_loss, 'Acc': train_acc, 'correct': f'{correct}/{total}'})
    pbar.close()

    return acc, mean_loss

def compute_auac(eps_list, acc_list):
    """Compute AUAC from a list of accuracy and the corresponding perturbation sizes.

    Args:
        eps_list (list of int): list of perturbation sizes.
        acc_list (list of float): accuracy obtained for the corresponding perturbation size.

    Returns:
        float: AUAC
    """
    auac = 0
    i = 0
    prev_eps = 0
    prev_acc = 0
    for eps, acc in zip(eps_list, acc_list):
        if i > 0:
            auac = ((acc + prev_acc)*(eps - prev_eps) / (2*eps)) + (auac*prev_eps)/eps
        prev_eps = eps
        prev_acc = acc
        i+=1
    
    return auac