# -*- coding: utf-8 -*-

import gudhi
import torch
from torch import nn
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from tools import gradient
from geometry import projection, uniform_resampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_skeleton_gpu(net, number, res, lenght, steps, div, resampling=False):
    
    # sampling points on the iso-surface
    iso = projection(net, number=number, iterations=10)
    if resampling:
        iso = uniform_resampling(iso, 1, 8, np.sqrt(1./number), 16/number)
        iso = projection(net, isopoints=iso, iterations=10)
    c = torch.linspace(0., lenght, res, device=device)
    L = []
    
    for k in range(steps):
        for d in tqdm(range(div)):
            #sdf and gradient
            l_iso = iso[d*number//div:(d+1)*number//div,:]
            sdf = net(l_iso)
            gra = nn.functional.normalize(-gradient(sdf, l_iso), dim=1)
            
            #lines generation
            pts = torch.mul(c[:,None,None], gra[None,:,:]) + l_iso
            sdf = net(pts)
            
            #inside-shape lines generation
            val,ind = torch.max(((sdf[1:,:,:]) > 0.).float(), dim=0) #find first point outside
            for i in range(l_iso.shape[0]):
                if k==0 and (ind[i] != 0 or val[i] == 0.):
                    L.append(d*number//div+i)
                if val[i] > 0.:
                    pts[:,i,:] = torch.mul(torch.linspace(0., c[ind[i]].item(), res, device=device)[:,None], gra[i][None,:]) + l_iso[i]
            sdf = net(pts)
            
            #finding smallest gradient's norm
            # grn = gradient(sdf, pts).norm(dim=2)
            # arg = torch.argmin(grn, dim=0)
            arg = torch.argmin(sdf, dim=0)
            for i in range(l_iso.shape[0]):
                iso[d*number//div+i] = pts[arg[i],i]
            del sdf, gra, pts, arg

    for d in range(div):
        l_iso = iso[d*number//div:(d+1)*number//div,:]
        sdf = net(l_iso)
        for i in range(l_iso.shape[0]):
            if sdf[i] > 0 and (d*number//div+i in L):
                L.remove(d*number//div+i)

    iso = iso.detach().cpu().numpy()
    iso = iso[np.array(L), :]
    
    #keep only points in unit cube
    keep = [i for i in range(iso.shape[0])]
    for i,p in enumerate(iso):
        if np.linalg.norm(p, np.inf) > 1.:
            keep.remove(i)
    iso = iso[keep]

    return iso


def reduce(m, reduce_radius):
    KD = KDTree(m)
    NN = KD.query_ball_tree(KD, reduce_radius)
    keep = [i for i in range(len(NN))]
    for i in range(len(NN)):
        if i in keep:
            for k in NN[i][1:]:
                if k>i and k in keep:
                    keep.remove(k)
    return m[keep]


def skpoints_to_mesh_gudhi(m, reduce_radius, alpha):
    # reducing vertices
    if reduce_radius != None:
        vertices = reduce(m, reduce_radius)
    else:
        vertices = m
    print(vertices.shape[0], "vertices")
    
    # building alpha-complex
    alpha_complex = gudhi.alpha_complex.AlphaComplex(points=vertices)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square = alpha**2)
    
    triangles = []
    bones = []
    sedges = set()
    
    # extracting simplices
    for s in simplex_tree.get_simplices():
        if len(s[0]) == 3:
            t = s[0]
            triangles.append(t)
            sedges.add(frozenset([t[0],t[1]]))
            sedges.add(frozenset([t[1],t[2]]))
            sedges.add(frozenset([t[2],t[0]]))
    
    for s in simplex_tree.get_simplices():
        if len(s[0]) == 2:
            e = s[0]
            if frozenset([e[0], e[1]]) not in sedges:
                bones.append(e)
    
    print(len(triangles), "triangles")
    print(len(sedges), "edges")
    print(len(bones), "bones")
    
    for t in triangles:
        for i in range(3):
            t[i] += 1
    for b in bones:
        for i in range(2):
            b[i] += 1

    v = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        v[i] = alpha_complex.get_point(i)

    return v, triangles, bones