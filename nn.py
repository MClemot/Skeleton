# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from tools import gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



### SIREN neural network ###

class Sine(nn.Module):
    def __init__(self, w0 = 30.):
        super().__init__()
        self.w0 = w0 
    def forward(self, x):
        return torch.sin(self.w0*x)   
    
class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        return self.activation(torch.nn.functional.linear(x, self.weight, self.bias))

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, skip = [], w0 = 30., w0_initial = 30., activation = None): 
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.skip = [i in skip for i in range(num_layers)]

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(SirenLayer(
                dim_in = layer_dim_in + (3 if self.skip[ind] else 0),
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = True,
                is_first = is_first,
                activation = activation
            ))
        self.last_layer = SirenLayer(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = True, activation = nn.Identity())

    def forward(self, x):
        i = x
        for k,layer in enumerate(self.layers):
            if not self.skip[k]:
                x = layer(x)
            else:
                x = layer(torch.concat((x,i), dim=-1))
        return self.last_layer(x)



### Utility ###

def dist_to_pc(pts, pc, grad, with_grad=False):
    dis = torch.cdist(pts,pc)
    mini = torch.min(dis, dim=1)
    s = torch.tensor([1 if torch.dot(grad[mini.indices[i]], pts[i]-pc[mini.indices[i]])>0 else -1 for i in range(pts.shape[0])], device=device)
    if with_grad:
        return mini.values * s, (pts - pc[mini.indices]) * s[:,None]
    else:
        return mini.values * s


### Losses ###

def sdf_loss_gradNorm(grad):
    return torch.nn.functional.mse_loss(grad.norm(dim=1), torch.ones_like(grad.norm(dim=1)))
    #return torch.nn.functional.l1_loss(grad.norm(dim=1), torch.ones_like(grad.norm(dim=1)))

def sdf_loss_zero(sdf):
    return torch.nn.functional.mse_loss(sdf, torch.zeros_like(sdf))
    #return torch.abs(sdf).mean()
    
def sdf_loss_align(grad, normals):
    return (1-nn.functional.cosine_similarity(grad, normals, dim = 1)).mean()
    
def sdf_loss_nonzero(sdf, alpha):
    #return torch.nn.functional.mse_loss(torch.exp(-alpha*torch.abs(sdf)), torch.zeros_like(sdf))
    return torch.exp(-alpha * torch.abs(sdf)).mean()
    
def sdf_loss_target(sdf, target_sdf):
    return torch.nn.functional.mse_loss(torch.flatten(sdf), target_sdf)


def sdf_loss_point_cloud(sdf, pts, normals):
    grad = gradient(sdf, pts)
    
    return (sdf_loss_zero(sdf) * 1e3 +
            sdf_loss_align(grad, normals) * 1e2 +
            sdf_loss_gradNorm(grad) * 2e2 )


def sdf_loss_others(sdf, grad):
    return (sdf_loss_gradNorm(grad) * 2e2)
    

def sdf_loss_hints(sdf, pts, target_sdf, target_grad):
    grad = gradient(sdf, pts)
    
    return (sdf_loss_target(sdf, target_sdf) * 3e3 +
            sdf_loss_gradNorm(grad)  * 5e1 +
            sdf_loss_align(grad, target_grad) * 2e2)


def sdf_loss_tv(grad, pts):
    grad2 = gradient(grad.norm(dim=1), pts).norm(dim=1)
    return grad2.mean() * 3e1


def sdf_loss_lineconsistency(net, pts, grad, sdf):
    N = 10
    t = torch.rand((pts.shape[0]*N), device=device)[:,None]
    return nn.functional.mse_loss(net(t*(pts.repeat(N,1)-torch.sign(sdf.repeat(N,1))*sdf.repeat(N,1)*nn.functional.normalize(grad.repeat(N,1), dim=1)) + (1-t)*pts.repeat(N,1)),
                                  (1-t)*sdf.repeat(N,1))
    
def sdf_loss_projection(net, pts, grad, sdf):
    proj = pts - torch.sign(sdf)*sdf*grad
    return nn.functional.mse_loss(net(proj), torch.zeros_like(net(proj)))

### Optimization ###

def optimize_neural_sdf(net, optim, pc, nc, batch_size, pc_batch_size, epochs, nb_hints, tv_ends, hints_ends):
    pts_hint = torch.rand((nb_hints, 3), device = device)*2-1
    pts_hint.requires_grad = True
    gt_sdf_hint, gt_grad_hint = dist_to_pc(pts_hint, pc, nc, with_grad=True)
    gt_sdf_hint = gt_sdf_hint.detach()
    gt_grad_hint = gt_grad_hint.detach()
    
    #backup lists to store and display the loss
    lpc, loth, lh, ltv = [], [], [], []
    
    for batch in tqdm(range(epochs)):
        pts_random = torch.rand((batch_size, 3), device = device)*2-1
        pts_random.requires_grad = True
      
        #predict the sdf for all points
        sample = torch.randint(pc.shape[0], (pc_batch_size,))
        sample_pc = pc if pc_batch_size == None else pc[sample]
        sample_nc = nc if pc_batch_size == None else nc[sample]
        pred_sdf_point_cloud = net(sample_pc)
        pred_sdf_hint = net(pts_hint)
        pred_sdf_random = net(pts_random)
        
        grad_random = gradient(pred_sdf_random, pts_random) 
      
        # compute and store the losses 
        loss_pc = sdf_loss_point_cloud(pred_sdf_point_cloud, sample_pc, sample_nc)
        loss_hint = sdf_loss_hints(pred_sdf_hint, pts_hint, gt_sdf_hint, gt_grad_hint) if batch < hints_ends else 0.
        loss_other = sdf_loss_others(pred_sdf_random, grad_random)
        loss_tv = sdf_loss_tv(grad_random, pts_random) if batch < tv_ends else 0.
        
        
        # append all the losses
        lpc.append(float(loss_pc))
        loth.append(float(loss_other))
        lh.append(float(loss_hint))
        ltv.append(float(loss_tv))
      
        # sum the losses of reach of this set of points
        loss = loss_pc + loss_other + loss_hint + 1/(1+(batch/500)**2)*loss_tv
        optim.zero_grad()
        loss.backward()
      
        optim.step()
      
        # display the result
        if False:#batch%50 == 49:
            plt.figure(figsize=(8,6))
            plt.yscale('log')
            plt.plot(lpc, label = f'Point cloud loss ({lpc[-1]})')
            plt.plot(loth, label = f'Other points loss ({loth[-1]})')
            plt.plot(lh, label = f'Hint points loss ({lh[-1]})')
            plt.plot(ltv, label = f'TV loss ({ltv[-1]})')
            plt.legend()
            plt.show()
        



def pretrain(dim_hidden, num_layers, skip, lr, batch_size, epochs):
    net = SirenNet(
        dim_in = 3,
        dim_hidden = dim_hidden,
        dim_out = 1,
        num_layers = num_layers,
        skip = skip,
        w0_initial = 30.,
        w0 = 30.,
        ).to(device)
    
    optim = torch.optim.Adam(lr=lr, params=net.parameters())
    
    lpc, loth = [], []
    
    try:
        for batch in tqdm(range(epochs)):
            pts_random = torch.rand((batch_size, 3), device = device)*2-1
            pts_random.requires_grad = True
            
            pred_sdf_random = net(pts_random)
            
            gt_sdf_random = torch.linalg.norm(pts_random, dim=1) - 0.5
            loss_pc = nn.functional.mse_loss(pred_sdf_random.flatten(), gt_sdf_random) * 1e1
            
            grad_random = gradient(pred_sdf_random, pts_random)    
            loss_other = sdf_loss_gradNorm(grad_random)
            
            # append all the losses
            lpc.append(float(loss_pc))
            loth.append(float(loss_other))
          
            # sum the losses of reach of this set of points
            loss = loss_pc + loss_other
            optim.zero_grad()
            loss.backward()
          
            optim.step()
          
            # display the result
            if False:#batch%50 == 49:
                plt.figure(figsize=(8,6))
                plt.yscale('log')
                plt.plot(lpc, label = f'Point cloud loss ({lpc[-1]})')
                plt.plot(loth, label = f'Other points loss ({loth[-1]})')
                plt.legend()
                plt.show()
    except KeyboardInterrupt:
        pass
    
    return net