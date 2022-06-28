# -*- coding: utf-8 -*-

import torch
import numpy as np
from pygel3d import hmesh

from tools import gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# IO 
# =============================================================================

def to_obj(pts, path, factor=1., tri = None, lines = None):
    f = open(path, "w")
    for p in pts:
        f.write("v {} {} {}\n".format(factor*p[0], factor*p[1], factor*p[2]))
    if tri != None:
        for t in tri:
            l = list(t)
            f.write("f {} {} {}\n".format(l[0], l[1], l[2]))
    if lines != None:
        for li in lines:
            l = list(li)
            f.write("l {} {}\n".format(l[0], l[1]))
    f.close()

def from_obj(path):
    f = open(path, "r")
    s = f.readline()
    L = []
    while s:
        t = s.split()
        L.append(np.array([float(t[1]), float(t[2]), float(t[3])]))
        s = f.readline()
    f.close()
    return np.array(L)

def load_mesh(s, normalize=True):
    m = hmesh.load(s)

    normals = []
    for v in m.vertices():
        normals.append(m.vertex_normal(v))
    normals = np.array(normals)
    
    M = 0
    for v in m.positions():
        M = max(M, np.linalg.norm(v, np.inf))
    
    pts_point_cloud, normal_point_cloud = (1/M if normalize else 1)*torch.tensor(m.positions(), device=device).float(), torch.tensor(normals, device=device).float()
    pts_point_cloud.requires_grad = True
    
    return pts_point_cloud, normal_point_cloud, M


def cube_point_cloud(c, pts_per_side, device):
    pts = torch.rand((pts_per_side, 6, 3))
    pts = pts * torch.tensor([[[c, c, 0], [c, 0, c], [0, c, c], [c, c, 0], [c, 0, c], [0, c, c]]]) + .5 * c * torch.tensor([[[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]) 
    normals = torch.ones_like(pts) * torch.tensor([[[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0]]])
    return pts.reshape(-1, 3).to(device), normals.reshape(-1, 3).to(device)



# =============================================================================
# sampling on mesh
# =============================================================================

def sample_mesh(s, N):
    m = hmesh.load(s)
    sample = []
    normals = []
    
    M = 0
    for v in m.positions():
        M = max(M, np.linalg.norm(v, np.inf))
    
    area = 0
    for t in m.faces():
        area += m.area(t)
    
    for t in m.faces():
        n = m.face_normal(t)
        ver = []
        for v in m.circulate_face(t, mode='v'):
            ver.append(v)
        r = m.area(t)/area * N
        if r<1:
            if np.random.rand() > r:
                continue
        num = int(np.ceil(r))
        r1 = np.random.rand(num)
        r2 = np.random.rand(num)
        loc = (np.ones_like(r1)-np.sqrt(r1))[:,None] * m.positions()[ver[0]][None,:]
        loc += (np.sqrt(r1)*(np.ones_like(r1)-r2))[:,None] * m.positions()[ver[1]][None,:]
        loc += (np.sqrt(r1)*r2)[:,None] * m.positions()[ver[2]][None,:]
        for p in loc:
            sample.append(p)
            normals.append(n)
    
    # to_obj(sample, "D:/test.obj")
    
    pc = 1/M * torch.tensor(np.array(sample), device=device).float()
    pc.requires_grad = True
    nc = torch.tensor(np.array(normals), device=device).float()
    return pc, nc



# =============================================================================
# geometry 
# =============================================================================

def projection(net, iterations=10, isopoints=None, number=1000, prune=False):
    #initializing iso-points if needed
    if isopoints == None:
        isopoints = torch.rand((number, 3), device = device)*1.9-.95
    if not isopoints.requires_grad:
        isopoints.requires_grad = True
    
    for it in range(iterations):
        isopoints_sdf = net(isopoints.float())
        grad = gradient(isopoints_sdf, isopoints)
        inv = grad / (torch.norm(grad, dim=1)**2)[:,None]
    
        isopoints = isopoints - inv * isopoints_sdf
        isopoints = isopoints.float()
    
        del grad, inv
        
    if prune:
        keep = [i for i in range(isopoints.shape[0])]
        for i,p in enumerate(isopoints):
            if torch.linalg.norm(p, torch.inf) > 1.:
                keep.remove(i)
        isopoints = isopoints[keep]
  
    return isopoints.float()

def uniform_resampling(isopoints, steps, K, alpha, sigma):
    for step in range(steps):
  
        #compute the distance between the points and the (K+1) nearest neighbors
        dismat = torch.cdist(isopoints, isopoints)
        _,knn = torch.topk(dismat, K+1, dim=1, largest=False)
    
        #delete the nearest neightbor which is always the point itself, to keep only the K true nearest neighbors
        knn = knn[:,1:]
    
        #compute with the KNN graph the wanted shift for each isopoint
        r = torch.zeros_like(isopoints)
        for i in range(isopoints.shape[0]):
            for k in range(K):
                r[i,:] = r[i,:] + torch.nn.functional.normalize(isopoints[knn[i,k],:] - isopoints[i,:], dim=0) * torch.exp(-dismat[i,knn[i,k]]**2/sigma)
        
        #move the isopoints
        isopoints = isopoints - alpha*r
  
    return isopoints.to(device)