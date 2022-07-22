# -*- coding: utf-8 -*-

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def hessian(f, x):
    g = gradient(f(x), x)
    h0 = gradient(g[:,0], x)[:,None,:]
    h1 = gradient(g[:,1], x)[:,None,:]
    h2 = gradient(g[:,2], x)[:,None,:]
    h = torch.cat((h0,h1,h2), dim=1)
    return h

def laplacian(f, x):  
    h = hessian(f,x)
    return torch.diagonal(h, dim1=1, dim2=2).sum(dim=1)

def laplacian2d(f, x):  
    g = gradient(f(x), x)
    h0 = gradient(g[:,0], x)[:,None,:]
    h1 = gradient(g[:,1], x)[:,None,:]
    h = torch.cat((h0,h1), dim=1)
    return torch.diagonal(h, dim1=1, dim2=2).sum(dim=1)
    
def gauss_curvature(f, x):
    g = gradient(f(x), x)
    h0 = gradient(g[:,0], x)[:,None,:]
    h1 = gradient(g[:,1], x)[:,None,:]
    h2 = gradient(g[:,2], x)[:,None,:]
    h = torch.cat((h0,h1,h2,g[:,None,:]), dim=1)
    g0 = torch.cat((g,torch.zeros((g.shape[0],1),device=device)), dim=1)
    h = torch.cat((h,g0[:,:,None]), dim=2)
    
    return - torch.linalg.det(h) / torch.linalg.norm(g, dim=1)**4

def area(f, eps):
    k = 10
    N = 100000
    s = 0
    for _ in range(k):
        p = torch.rand((N,3), device=device)*2-1
        sdf = f(p)
        sel = (torch.abs(sdf) <= eps).float()
        s += sel.sum().item()
    
    return 8*s/N/(2*eps)/k
    