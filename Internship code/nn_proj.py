# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

from nn import SirenNet, Relu
from geometry import sample_mesh, to_obj
from tools import gradient
from display import display_sdfColor, display_angle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Losses
# =============================================================================


def proj_loss_idempot(net, proj):
    return nn.functional.mse_loss(net(proj), proj)

def proj_loss_lineconsistency(net, rand, proj):
    N = 10
    t = torch.rand((rand.shape[0]*N), device=device)[:,None]
    return nn.functional.mse_loss(net(t*proj.repeat(N,1)+(1-t)*rand.repeat(N,1)), proj.repeat(N,1))

def proj_loss_normal(pc, proj_pc, nc):
    cs = nn.functional.cosine_similarity(pc-proj_pc, nc, dim=1)
    # return (1 - cs).mean()
    return torch.exp(-5*cs).mean()

def proj_loss_align(grad, normals):
    return (1-nn.functional.cosine_similarity(grad, normals, dim = 1)).mean()

def proj_loss_gradient(rand, proj):
    df = torch.linalg.norm(rand-proj, dim=1)
    grad = gradient(df, rand)
    return nn.functional.mse_loss(torch.linalg.norm(grad, dim=1), torch.ones((rand.shape[0]), device=device))

def proj_loss_rank(net, proj):
    diff0 = gradient(net(proj)[:,0], proj)
    diff1 = gradient(net(proj)[:,1], proj)
    diff2 = gradient(net(proj)[:,2], proj)
    diff=torch.stack((diff0,diff1,diff2),dim=2)
    rank = torch.linalg.matrix_rank(diff)


# =============================================================================
# Optimization
# =============================================================================
    

def optimize_neural_proj(net, optim, pc, nc, batch_size, pc_batch_size, epochs, nb_hints, mode='skeleton'):
    
    #bakup lists to store and display the loss
    l1, l2, l3, l4, l5, l6, l7 = [], [], [], [], [], [], []
    
    pc_cpu = pc.detach().cpu().numpy()
    KDT = KDTree(pc_cpu)
    pts_hint = torch.rand((nb_hints, 3), device = device)*2-1
    pts_hint_cpu = pts_hint.detach().cpu().numpy()
    NNv, NNi = KDT.query(pts_hint_cpu)
    gt_proj_hint = pc[NNi].detach()
    
    s = torch.tensor([1 if torch.dot(nc[NNi[i]], pts_hint[i]-pc[NNi[i]])>0 else -1 for i in range(nb_hints)], device=device)
    gt_sdf_hint = torch.tensor(NNv,device=device).float() * s
    
    for batch in range(epochs):
        loss_idempot,loss_line,loss_pc,loss_normal,loss_gradient,loss_hint,loss_sdf = 0,0,0,0,0,0,0
        
        pts_random = torch.rand((batch_size, 3), device = device)*2-1
        pts_random.requires_grad = True
      
        sample = torch.randint(pc.shape[0], (pc_batch_size,))
        sample_pc = pc if pc_batch_size == None else pc[sample]
        sample_nc = nc if pc_batch_size == None else nc[sample]

        proj_random = net(pts_random)
        proj_pc = net(sample_pc)
        proj_hint = net(pts_hint)
      
        # compute and store the losses
        if mode=='skeleton':
            loss_idempot = proj_loss_idempot(net, proj_pc) * 10
            loss_line = proj_loss_lineconsistency(net, sample_pc[:5000,:], proj_pc[:5000,:]) * 10
            loss_pc = torch.exp(-10*torch.linalg.norm(sample_pc - proj_pc, dim=1)).mean() * 1
            loss_normal = proj_loss_normal(sample_pc, proj_pc, sample_nc) * 1
            loss_gradient = 0#proj_loss_gradient(pts_random, proj_random) * 1
            
        elif mode=='surface':
            loss_idempot = proj_loss_idempot(net, proj_random) * 10
            loss_line = proj_loss_lineconsistency(net, pts_random[:5000,:], proj_random[:5000,:]) * 100
            loss_pc = nn.functional.mse_loss(proj_pc, sample_pc) * 100
            loss_gradient = 0#proj_loss_gradient(pts_random, proj_random) * 1
            loss_hint = nn.functional.mse_loss(proj_hint, gt_proj_hint) * 10
            
        elif mode=='surface4':
            loss_idempot = proj_loss_idempot(lambda x:net(x)[:,:3], proj_random[:,:3]) * 10
            loss_line = proj_loss_lineconsistency(lambda x:net(x)[:,:3], pts_random[:10000,:3], proj_random[:10000,:3]) * 100
            loss_pc = nn.functional.mse_loss(proj_pc[:,:3], sample_pc) * 100
            loss_normal = proj_loss_align(gradient(proj_pc[:,3], sample_pc), sample_nc) * 10
            loss_gradient = 0#(proj_loss_gradient(pts_random, proj_random[:,:3])
            loss_gradient += nn.functional.mse_loss(torch.norm(gradient(proj_random[:,3], pts_random), dim=1), torch.ones(batch_size, device=device)) * 10
            loss_hint = nn.functional.mse_loss(proj_hint, torch.concat((gt_proj_hint, gt_sdf_hint[:,None]), dim=1)) * 100
            loss_sdf = nn.functional.mse_loss(torch.norm(proj_random[:,:3]-pts_random, dim=1), torch.abs(proj_random[:,3])) * 10
            
        elif mode=="fullhints":
            pts_hint_cpu = pts_random.detach().cpu().numpy()
            _, NN = KDT.query(pts_hint_cpu, workers=4)
            gt_proj_hint = pc[NN].detach()
            
            loss_pc = nn.functional.mse_loss(proj_pc, sample_pc)
            loss_hint = nn.functional.mse_loss(proj_random, gt_proj_hint)
        
        # append all the losses
        l1.append(float(loss_idempot))
        l2.append(float(loss_line))
        l3.append(float(loss_pc))
        l4.append(float(loss_normal))
        l5.append(float(loss_gradient))
        l6.append(float(loss_hint))
        l7.append(float(loss_sdf))
      
        # sum the losses of reach of this set of points
        loss = loss_idempot + loss_pc + loss_line + loss_normal + loss_gradient + loss_hint + loss_sdf
        
        optim.zero_grad()
        loss.backward()
      
        optim.step()
      
        # display the result
        if batch%50 == 49:
            plt.figure(figsize=(8,6))
            plt.yscale('log')
            plt.plot(l1, label = f'Idempotence loss ({l1[-1]})')
            plt.plot(l2, label = f'Line loss ({l2[-1]})')
            plt.plot(l3, label = f'Point cloud loss ({l3[-1]})')
            plt.plot(l4, label = f'Normal loss ({l4[-1]})')
            plt.plot(l5, label = f'Gradient loss ({l5[-1]})')
            plt.plot(l6, label = f'Hint loss ({l6[-1]})')
            plt.plot(l7, label = f'Consistant SDF loss ({l7[-1]})')
            plt.legend()
            plt.show()
            
            
def pretrain(dim_hidden, num_layers, skip, lr, batch_size, epochs):
    net = SirenNet(
        dim_in = 3,
        dim_hidden = dim_hidden,
        dim_out = 4,
        num_layers = num_layers,
        skip = skip,
        w0_initial = 30.,
        w0 = 30.,
        activation = Relu()
        ).to(device)
    
    optim = torch.optim.Adam(lr=lr, params=net.parameters())
    
    lproj, lsdf = [], []
    
    try:
        for batch in range(epochs):
            pts_random = torch.rand((batch_size, 3), device = device)*2-1
            pts_random.requires_grad = True
            
            pred_random = net(pts_random)
            
            norm = torch.linalg.norm(pts_random, dim=1)
            gt_proj_random = nn.functional.normalize(pts_random) * 0.5
            
            loss_sdf = nn.functional.mse_loss(pred_random[:,3].flatten(), norm-0.5)
            loss_proj = nn.functional.mse_loss(pred_random[:,:3], gt_proj_random)
            
            # append all the losses
            lproj.append(float(loss_proj))
            lsdf.append(float(loss_sdf))
          
            # sum the losses of reach of this set of points
            loss = loss_proj + loss_sdf
            optim.zero_grad()
            loss.backward()
          
            optim.step()
          
            # display the result
            if batch%50 == 49:
                plt.figure(figsize=(8,6))
                plt.yscale('log')
                plt.plot(lproj, label = f'Projection loss ({lproj[-1]})')
                plt.plot(lsdf, label = f'SDF loss ({lsdf[-1]})')
                plt.legend()
                plt.show()
    except KeyboardInterrupt:
        pass
    
    return net

            
s = "helice"
mode = 'surface4'
pc, nc = sample_mesh("Objects/{}.obj".format(s), 100000)

# net = pretrain(128, 8, [], 2e-5, 50000, 4000)
# torch.save(net, "Pretrained/pretrainedproj_{}_{}".format(128, 8))

# net = torch.load("Pretrained/pretrainedproj_{}_{}".format(128, 8))

# optim = torch.optim.Adam(lr=2e-5, params=net.parameters())

# try:
#     optimize_neural_proj(net, optim, pc, nc,
#                           batch_size=25000, pc_batch_size=25000,
#                           epochs=2000, nb_hints=10000,
#                           mode=mode)
# except KeyboardInterrupt:
#     pass

# torch.save(net, "Networks/netproj{}_{}.net".format(mode, s))

net = torch.load("Networks/netproj{}_{}.net".format(mode, s))
pc = torch.rand((10000, 3), device = device)*2-1
pts_random = net(pc)[:,:3].detach().cpu().numpy()
to_obj(pts_random, "D:/proj__{}.obj".format(s))

for z in np.linspace(.1, -.4, 1):
    f = lambda x:torch.concat((x[:,:1],torch.zeros((x.shape[0],1), device=device)+z,x[:,1:]), dim=1)
    g = lambda x:net(f(x))[:,:3]-f(x)
    h = lambda x:torch.linalg.norm(g(x), dim=1)
    k = lambda x:torch.atan2(g(x)[:,0], g(x)[:,2])
    plt.figure()
    display_sdfColor(h, 500)
    if mode=="surface4":
        l = lambda x:net(f(x))[:,3]
        plt.figure()
        display_sdfColor(l, 500)
    plt.figure()
    display_angle(k, 500)
    

# plt.figure(figsize = (40, 40))
# ax = plt.axes(projection ="3d")
# ax.scatter3D(pts_random[:,0], pts_random[:,1], pts_random[:,2], c=torch.linalg.norm(pc.detach().cpu(),dim=1), cmap="hot")
# plt.show()

# N=10000
# plt.figure(figsize = (40, 40))
# ax = plt.axes(projection ="3d")
# pc = pc.detach().cpu().numpy()
# L = [[i+1,i+1+N] for i in range(N)]
# for i in range(N):
#     ax.plot3D([pc[i,0],pts_random[i,0]], [pc[i,1],pts_random[i,1]], [pc[i,2],pts_random[i,2]])
# plt.show()
# to_obj(np.concatenate([pc[:N], pts_random[:N]]), "D:/gt_proj.obj", lines=L)

# img = sphere_tracing_gpu(lambda x:net(x)[:,3:], 250, 0*np.pi/6, 3*np.pi/6, 1., 3., 1., .02, np.array([.6,.8,.3]))
# plt.figure(figsize=(10,10))  
# # plt.imshow(img)
# plt.imshow(np.rot90(img))
