# @copyright CEA-LIST/DIASI/SIALV/LVA (2021)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import numpy as np
import torch
from torch import nn

def coral(source, target):
    """compute the coral of 2 batch
    
    Arguments:
        source {Tensor} -- batch source
        target {Tensor} -- batch target
    
    Returns:
        loss -- coral loss of the two batch
    """
    d = source.size(1)  # dim vector

    xm = torch.mean(source, dim=0, keepdim=True) - source
    xc = torch.matmul(torch.t(xm), xm)
    
    xmt = torch.mean(target, dim=0, keepdim=True) - target
    xct = torch.matmul(torch.t(xmt), xmt)
    
    loss = torch.mean(torch.abs(xc - xct))

    return loss

def mmd(source, target):
    """compute the mmd of 2 batch
    
    Arguments:
        source {Tensor} -- batch source
        target {Tensor} -- batch target
    
    Returns:
        loss -- mmd loss of the two batch
    """
    xm = torch.mean(source, dim=0)
    xmt = torch.mean(target, dim=0)
    loss = torch.mean(torch.abs(xm - xmt))
    
    return loss

class AdversarialSinkhornDivergence(nn.Module):
    """
    Loss for Sikhorn Adversarial Training. Computes cross entropy of adversarial batch
    and Sinkhorn Divergence between the two batch.
    """
    def __init__(self, args, reduction='sum', num_class=10, epsilon=0.1, max_iter=50):
        super(AdversarialSinkhornDivergence, self).__init__()
        self.reduction = reduction
        self.num_class = num_class
        self.args = args
        self.eps = epsilon
        self.max_iter = max_iter

    def forward(self, outputs_clean, outputs_adv, target):

        crossentropy = nn.CrossEntropyLoss()
        
        # Cross entropy on adversarial batch
        loss_adv = crossentropy(outputs_adv, target)
        if self.reduction == 'sum':
            loss_adv = torch.sum(loss_adv) 
        if self.reduction == 'mean':
            loss_adv = torch.mean(loss_adv)

        sink = SinkhornDistance(eps=self.eps, max_iter=self.max_iter, args=self.args)

        # Compute the divergence
        dist, _, _ = sink(outputs_clean, outputs_adv)
        dist_adv, _, _ = sink(outputs_adv, outputs_adv)
        dist_clean, _, _ = sink(outputs_clean, outputs_clean)
        dist = dist - 0.5*(dist_adv + dist_clean)
         
        loss_sink = dist
        
        return loss_adv, loss_sink

class AdversarialDomainAdaptation(nn.Module):
    """
    Loss for Adversarial Training with Domain Adaptation.
    Computes cross entropy of clean batch, cross entropy of adversarial batch, mmd, coral between the two batch.
    """
    def __init__(self, device, reduction='sum', intra=False, num_class=10):
        super(AdversarialDomainAdaptation, self).__init__()
        self.reduction = reduction
        self.intra = intra
        self.num_class = num_class
        self.device = device

    def forward(self, outputs_clean, outputs_adv, target):

        crossentropy = nn.CrossEntropyLoss()

        loss_clean = crossentropy(outputs_clean, target)
        if self.reduction == 'sum':
            loss_clean = torch.sum(loss_clean) 
        if self.reduction == 'mean':
            loss_clean = torch.mean(loss_clean)
            
        loss_adv = crossentropy(outputs_adv, target)
        if self.reduction == 'sum':
            loss_adv = torch.sum(loss_adv) 
        if self.reduction == 'mean':
            loss_adv = torch.mean(loss_adv)

        loss_mmd = mmd(outputs_clean, outputs_adv)
        loss_coral = coral(outputs_clean, outputs_adv)
        
        return loss_clean, loss_adv, loss_mmd, loss_coral
    
# Adapted from https://github.com/dfdazac/wassdistance

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, args, p=2, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.args = args
        self.p = p

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y) # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        if self.args.gpu is not None:
            C = C.cuda(self.args.gpu, non_blocking=True)
            mu = mu.cuda(self.args.gpu, non_blocking=True)
            nu = nu.cuda(self.args.gpu, non_blocking=True)
            u = u.cuda(self.args.gpu, non_blocking=True)
            v = v.cuda(self.args.gpu, non_blocking=True)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1