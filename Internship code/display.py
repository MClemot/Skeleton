# -*- coding: utf-8 -*-

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from tools import gradient, hessian, laplacian

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def display_sdfColor(f, resolution, z, axis='z'):
    """
    displays the values of the function f, evaluated over a regular grid defined between -1 and 1 and of resolution (resolution x resolution)
    """
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2).reshape(-1, 2)
    if axis == 'x':
        coords = torch.concat((torch.zeros((coords.shape[0],1), device=device)+z, coords), dim=1)
    elif axis == 'y':
        coords = torch.concat((coords[:,:1], torch.zeros((coords.shape[0],1), device=device)+z, coords[:,1:]), dim=1)
    elif axis == 'z':
        coords = torch.concat((coords, torch.zeros((coords.shape[0],1), device=device)+z), dim=1)
    coords.requires_grad = True
    sdf = f(coords).reshape(resolution, resolution)
    numpy_sdf = sdf.detach().cpu().numpy()

    eps = 0.005
    numpy_sdf_max = np.ones(numpy_sdf.shape)-np.maximum(numpy_sdf,np.zeros(numpy_sdf.shape))
    numpy_sdf_max = numpy_sdf_max - np.multiply(numpy_sdf_max, np.multiply(numpy_sdf<=eps, numpy_sdf>=-eps))
    numpy_sdf_min = np.ones(numpy_sdf.shape)-np.maximum(-numpy_sdf,np.zeros(numpy_sdf.shape))
    numpy_sdf_min = numpy_sdf_min - np.multiply(numpy_sdf_min, np.multiply(numpy_sdf<=eps, numpy_sdf>=-eps))
    numpy_sdf_both = np.ones(numpy_sdf.shape)-np.maximum(numpy_sdf,np.zeros(numpy_sdf.shape))-np.maximum(-numpy_sdf,np.zeros(numpy_sdf.shape))
    numpy_sdf_both = numpy_sdf_both - np.multiply(numpy_sdf_both, np.multiply(numpy_sdf<=eps, numpy_sdf>=-eps))

    plt.axis('off')
    plt.imshow(np.concatenate([numpy_sdf_min[:,:,np.newaxis],numpy_sdf_both[:,:,np.newaxis],numpy_sdf_max[:,:,np.newaxis]], axis = 2) )
    plt.contour(numpy_sdf, 30, colors='k', linewidths=.4, linestyles='solid')
    plt.show()

def display_grad(f, resolution, z, axis='z'):
    """
    displays the norm of the gradient of the function f, evaluated at points of a regular grid defined between -1 and 1 and of resolution (resolution x resolution)
    """
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2).reshape(-1, 2)
    if axis == 'x':
        coords = torch.concat((torch.zeros((coords.shape[0],1), device=device)+z, coords), dim=1)
    elif axis == 'y':
        coords = torch.concat((coords[:,:1], torch.zeros((coords.shape[0],1), device=device)+z, coords[:,1:]), dim=1)
    elif axis == 'z':
        coords = torch.concat((coords, torch.zeros((coords.shape[0],1), device=device)+z), dim=1)
    coords.requires_grad = True
    sdf = f(coords)
    grad = gradient(sdf, coords).norm(dim = 1).detach().cpu().numpy().reshape(resolution, resolution)
    
    plt.axis('off')
    plt.imshow(grad, cmap = "nipy_spectral", vmin = 0., vmax = 1.5)  # 0.25, 1.25
    plt.colorbar()
    plt.show()

def display_gradgrad(f, resolution, z, axis='z'):
    """
    displays the norm of the gradient of the norm of the gradient of the function f, evaluated at points of a regular grid defined between -1 and 1 and of resolution (resolution x resolution)
    """
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2).reshape(-1, 2)
    if axis == 'x':
        coords = torch.concat((torch.zeros((coords.shape[0],1), device=device)+z, coords), dim=1)
    elif axis == 'y':
        coords = torch.concat((coords[:,:1], torch.zeros((coords.shape[0],1), device=device)+z, coords[:,1:]), dim=1)
    elif axis == 'z':
        coords = torch.concat((coords, torch.zeros((coords.shape[0],1), device=device)+z), dim=1)
    coords.requires_grad = True
    sdf = f(coords)
    grad = gradient(sdf, coords).norm(dim=1)
    grad2 = gradient(grad, coords).norm(dim=1).detach().cpu().numpy().reshape(resolution, resolution)
    
    plt.axis('off')
    plt.imshow(grad2, cmap = "plasma")#, vmin = 0, vmax = 20)
    plt.colorbar()
    plt.show()
    
def display_laplacian(f, resolution, z, axis='z'):
    """
    displays the norm of the gradient of the norm of the gradient of the function f, evaluated at points of a regular grid defined between -1 and 1 and of resolution (resolution x resolution)
    """
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2).reshape(-1, 2)
    if axis == 'x':
        coords = torch.concat((torch.zeros((coords.shape[0],1), device=device)+z, coords), dim=1)
    elif axis == 'y':
        coords = torch.concat((coords[:,:1], torch.zeros((coords.shape[0],1), device=device)+z, coords[:,1:]), dim=1)
    elif axis == 'z':
        coords = torch.concat((coords, torch.zeros((coords.shape[0],1), device=device)+z), dim=1)
    coords.requires_grad = True
    
    l = laplacian(f, coords).detach().cpu().numpy().reshape(resolution, resolution)
    
    plt.imshow(l, cmap = "plasma")#, vmin = 0, vmax = 20)
    plt.colorbar()
    plt.show()
    
def display_eig(f, resolution, z):
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2).reshape(-1, 2)
    coords = torch.concat((coords, torch.zeros((coords.shape[0],1), device=device)+z), dim=1)
    coords.requires_grad = True
    h = hessian(f, coords)
    eig = torch.linalg.eigvalsh(h)
    for i in range(3):
        plt.figure(figsize=(6,6))
        plt.imshow(eig[:,i].detach().cpu().numpy().reshape(resolution, resolution), cmap = "CMRmap", vmin = torch.min(eig), vmax = torch.max(eig))
        plt.colorbar()

def display_extrema(f, resolution, z):
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2).reshape(-1, 2)
    coords = torch.concat((coords, torch.zeros((coords.shape[0],1), device=device)+z), dim=1)
    coords.requires_grad = True
    h = hessian(f, coords)
    eig = torch.linalg.eigvalsh(h)
    sdf = f(coords)
    res = torch.where(torch.flatten(sdf > 0),
                      torch.maximum(eig[:,0], torch.zeros_like(eig[:,0])),
                      torch.minimum(eig[:,2], torch.zeros_like(eig[:,2])))
    plt.figure(figsize=(6,6))
    b = max(abs(torch.min(res)), abs(torch.max(res)))
    plt.imshow(res.detach().cpu().numpy().reshape(resolution, resolution), cmap = "seismic", vmin = -b, vmax = b)
    plt.colorbar()
    
def display_angle(f, resolution):
    x = torch.linspace(-1, 1, resolution, device = device)
    xy = [x]*2 #equivalent to [x, x]
    X, Y = torch.meshgrid(xy, indexing = 'xy')
    coords = torch.stack([X, Y], dim=2).reshape(-1, 2)
    angle = f(coords).detach().cpu().numpy().reshape(resolution, resolution)
    angle = (angle+np.pi)/(2*np.pi)
    clr = colors.hsv_to_rgb(np.concatenate([angle[:,:,None], np.ones((resolution, resolution, 2))], axis=2))
 
    plt.imshow(clr)
    plt.axis('off')
    plt.show()