# -*- coding: utf-8 -*-

import gudhi
import torch
from torch import nn
import numpy as np
from scipy.spatial import Delaunay, KDTree
import matplotlib.pyplot as plt

from tools import gradient
from geometry import to_obj, projection, uniform_resampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_skeleton_gpu(net, orientation, number, res, lenght, steps, div, s, resampling=False):
    iso = projection(net, number=number, iterations=10)
    if resampling:
        iso = uniform_resampling(iso, 1, 8, np.sqrt(1./number), 16/number)
        iso = projection(net, isopoints=iso, iterations=10)
    c = torch.linspace(0., lenght, res, device=device)
    L = []
    
    for k in range(steps):
        for d in range(div):
            #sdf and gradient
            l_iso = iso[d*number//div:(d+1)*number//div,:]
            sdf = net(l_iso)
            gra = nn.functional.normalize(orientation * gradient(sdf, l_iso), dim=1)
            
            #lines generation
            pts = torch.mul(c[:,None,None], gra[None,:,:]) + l_iso
            sdf = net(pts)
            
            #in-shape lines generation
            val,ind = torch.max(((orientation*sdf[1:,:,:]) < 0.).float(), dim=0) #find first point outside
            for i in range(l_iso.shape[0]):
                if k==0 and (ind[i] != 0 or val[i] == 0.):
                    L.append(d*number//div+i)
                if val[i] > 0.:
                    pts[:,i,:] = torch.mul(torch.linspace(0., c[ind[i]].item(), res, device=device)[:,None], gra[i][None,:]) + l_iso[i]
            sdf = net(pts)
            
            #finding smallest gradient if we search the lowest gradient's norm...
            grn = gradient(sdf, pts).norm(dim=2)
            arg = torch.argmin(grn, dim=0)
            # ...or the lowest SDF
            # arg = torch.argmin(sdf, dim=0)
            for i in range(l_iso.shape[0]):
                iso[d*number//div+i] = pts[arg[i],i]
            del sdf, gra, pts, grn, arg

    for d in range(div):
        l_iso = iso[d*number//div:(d+1)*number//div,:]
        sdf = net(l_iso)
        for i in range(l_iso.shape[0]):
            if orientation*sdf[i] < 0 and (d*number//div+i in L):
                L.remove(d*number//div+i)

    iso = iso.detach().cpu().numpy()
    iso = iso[np.array(L), :]
    
    #keep only points in unit cube
    keep = [i for i in range(iso.shape[0])]
    for i,p in enumerate(iso):
        if np.linalg.norm(p, np.inf) > 1.:
            keep.remove(i)
    iso = iso[keep]
    
    # to_obj(iso, 'Skeletons/sk_n{}_res{}_s{}_{}.obj'.format(number, res, steps, s))
    return iso


def find_skeleton_elasticity(net, number, c_sdf, c_repulsion, c_elasticity, k_0):
    iso = torch.rand((number, 3), device = device)*1.9-.95
    iso = nn.Parameter(iso)
    
    optim = torch.optim.Adam([iso], lr=.01)
    
    for i in range(200):       
        loss_sdf = torch.mean(net(iso)) * c_sdf
        
        KD = KDTree(iso.detach().cpu().numpy())
        kNN = KD.query(iso.detach().cpu().numpy(), k=9)[1][:,1:]
        diff = iso[:,None,:].repeat(1,8,1) - iso[kNN]
        dist = torch.linalg.norm(diff, dim=2)

        loss_repulsion = torch.mean(1/dist) * c_repulsion
        loss_elasticity = torch.mean((dist[:,:2]-k_0)**2) * c_elasticity
        
        loss = loss_sdf + loss_repulsion + loss_elasticity
        print(loss_sdf.item(), loss_repulsion.item(), loss_elasticity.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

    to_obj(iso, "D:/elas.obj")
    return iso.detach().cpu().numpy()


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
    
    alpha_complex = gudhi.alpha_complex.AlphaComplex(points=vertices)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square = alpha**2)
    
    triangles = []
    bones = []
    sedges = set()
    
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


def coverage_skeleton(net, pc, skpts, reduce_radius):
    # reducing vertices
    skpts = torch.tensor(reduce(skpts, reduce_radius), device=device).float()
    n = skpts.shape[0]
    print(n, "vertices")
    
    delta = 0.06
    sdf = -net(skpts)
    L = torch.cdist(skpts, pc)
    D = (L <= 1.3*(sdf + 0*delta))
    
    v = torch.ones((n), device=device)
    res = torch.matmul(v, D.float()).detach().cpu().numpy()
    plt.hist(res, bins=50)
    
    if not torch.all(torch.matmul(v, D.float()) >= 1.):
        L = []
        for i in range(pc.shape[0]):
            if res[i] == 0.:
                L.append(i)
        return None
    
    for k in range(4*n):
        i = np.random.randint(0,n)
        v[i] = 0
        if not torch.all(torch.matmul(v, D.float()) >= 1.):
            v[i] = 1
    
    L = []
    for i in range(n):
        if v[i]==1:
            L.append(i)
            
    print(len(L))
            
    skpts = skpts[L]
    to_obj(skpts, "D:/coverage.obj")
    
    return skpts.detach().cpu().numpy()

def quasinormal_skeleton(net, number):
    
    iso = projection(net, number=number, iterations=10)
    grad = gradient(net(iso), iso).detach()
    iso = iso.detach()
    sk = torch.clone(iso)
    sk = sk - 0.001*grad
    sk.requires_grad = True
    sk = nn.Parameter(sk)
    
    optim = torch.optim.Adam([sk], lr=.01)
    
    for i in range(100):       
        loss_sdf = torch.mean(net(sk))
        loss_dir = (-nn.functional.cosine_similarity(grad, iso-sk, dim = 1)).mean()
        
        loss = loss_sdf + .00001*loss_dir
        print(loss_sdf.item(), loss_dir.item())

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    return sk