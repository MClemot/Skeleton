# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm

from tools import gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def intersect_unit_cube(p, ray):
    I = []
    for c in [-1,1]:
        for i in range(3):
            t = (c-p[i])/ray[i]
            q = p+ray*t
            if np.linalg.norm(q, np.inf) < 1.0001 and t>0:
                I.append((t,q))
    if len(I)==0:
        return None
    else:
        return min(I, key=lambda x:x[0])[1]
            

def sphere_tracing_gpu(f, N, phi, theta, r, foc, wfoc, eps, light):
    """
    f : implicit function to sphere-trace
    N : resolution of the image
    (r,phi,theta) : spherical coordinates of the camera, always looking at the origin
    foc : focal length of the camera
    eps : precision parameter for stooping the sphere-tracing
    light : direction of the light for shading
    """
  
    er   = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    eth  = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]);
    ephi = np.array([-np.sin(phi), np.cos(phi), 0.]);
  
    focus = (r+foc)*er
  
    img = np.zeros((N,N,3))
  
    for i in tqdm(range(N)):
        L = []
        for j in range(N):
            proj = r*er + wfoc*(2*i-N)/N*ephi + wfoc*(2*j-N)/N*eth
            ray = proj - focus
            
            ray = ray / np.linalg.norm(ray)
      
            p = intersect_unit_cube(focus, ray)
            if p is None:
                continue
            else:
                L.append((i,j,ray,p))
  
        ray = torch.zeros((len(L), 3), device=device)
        p = torch.zeros((len(L), 3), device=device)
        d = torch.zeros((len(L)), device=device)
        for k,e in enumerate(L):
            ray[k] = torch.from_numpy(e[2])
            p[k] = torch.from_numpy(e[3])
      
        for _ in range(30):
            d = f(p)
            p = p + ((torch.linalg.norm(p, dim=1, ord=torch.inf) <= 1)[:,None]*d)*ray
            
        # out = torch.linalg.norm(p, dim=1, ord=torch.inf) > 1
        # cvg = f(p) < eps
        # print(torch.logical_or(out, cvg).all())
        
        grad = gradient(f(p), p)
        
        for k in range(len(L)):
            if d[k] < eps:
                normal = grad[k].cpu().detach().numpy()
                img[L[k][1], L[k][0]] = max(0., np.dot(normal / np.linalg.norm(normal), light / np.linalg.norm(light)))
          
    return img#.transpose((1, 0, 2))