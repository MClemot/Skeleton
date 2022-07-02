# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import os
import random

from nn import Sine, SirenLayer, sdf_loss_point_cloud, sdf_loss_others, sdf_loss_hints_old, dist_to_pc
from geometry import load_mesh, from_ply, to_obj, sample_mesh
from tools import gradient
from display import display_sdfColor, display_grad
from render import sphere_tracing_gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ShapeSpaceNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, num_shapes, dim_latent, skip = [], w0 = 30., w0_initial = 30., activation = None): 
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        
        self.num_shapes = num_shapes
        self.dim_latent = dim_latent
        latent = torch.zeros((num_shapes, dim_latent))
        self.latent = nn.Parameter(latent)
        
        self.skip = [i in skip for i in range(num_layers)]

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = (dim_in + dim_latent) if is_first else dim_hidden

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
    
    
    
### Optimization ###

def optimize_ss_neural_sdf(net, optim,
                           dataset, dataset_batch_size, dataset_epochs,
                           batch_size, epochs, num_hints, batch_pc_maxsize):
    
    dataset_size = len(dataset)
    
    for batch_ind in range(dataset_epochs):
        batch = np.random.choice(dataset_size, size=dataset_batch_size, replace=False)
        
        print(batch)
        
        meshes_pc, meshes_nc, gt_sdf_hint = [], [], []
        pts_hint = torch.rand((num_hints, 3), device = device)*2-1
        pts_hint.requires_grad = True
        pts_hint_cpu = pts_hint.detach().cpu().numpy()
        
        for b in batch:
            pc,nc,_ = load_mesh(dataset[b])
            # pc, nc = from_ply(dataset[b])
            # pc, nc = torch.tensor(pc, device=device).float(), torch.tensor(nc, device=device).float()
            # pc.requires_grad = True
            meshes_pc.append(pc)
            meshes_nc.append(nc)
            
            print(b, pc.shape[0])
            
            pc_cpu = pc.detach().cpu().numpy()
            KDT = KDTree(pc_cpu)
            NNv, NNi = KDT.query(pts_hint_cpu)
            s = torch.tensor([1 if torch.dot(nc[NNi[i]], pts_hint[i]-pc[NNi[i]])>0 else -1 for i in range(num_hints)], device=device)
            gt_sdf_hint.append((torch.tensor(NNv,device=device).float() * s).detach())
        
        l = []
        llat, lpc, lrd, lh = [], [], [], []
        
        for minibatch in range(epochs):
            
            pts_random = torch.rand((batch_size, 3), device = device)*2-1
            pts_random.requires_grad = True
            
            loss_latent = .01 * torch.sum(torch.linalg.norm(net.latent, dim=1))
            loss_random = 0
            loss_pc = 0
            loss_hint = 0
            
            for i,b in enumerate(batch):
                zb = net.latent[b,:]
                if meshes_pc[i].shape[0] <= batch_pc_maxsize:
                    pc = meshes_pc[i]
                else:
                    pc = meshes_pc[i][torch.randint(0,meshes_pc[i].shape[0],(batch_pc_maxsize,))]
                
                
                sdf_random = net(torch.concat((pts_random, zb.repeat((batch_size,1))), dim=1))
                sdf_pc = net(torch.concat((pc, zb.repeat((pc.shape[0],1))), dim=1))
                sdf_hint = net(torch.concat((pts_hint, zb.repeat((num_hints,1))), dim=1))
                
                # sdf_random = net(torch.concat((pts_random, torch.zeros((batch_size,32), device=device)), dim=1))
                # sdf_pc = net(torch.concat((pc, torch.zeros((2000,32), device=device)), dim=1))
                # sdf_hint = net(torch.concat((pts_hint, torch.zeros((2000,32), device=device)), dim=1))
                
                grad_random = gradient(sdf_random, pts_random)
                loss_random += sdf_loss_others(sdf_random, grad_random)
                loss_pc += sdf_loss_point_cloud(sdf_pc, meshes_pc[i], meshes_nc[i])
                loss_hint += sdf_loss_hints_old(sdf_hint, pts_hint, gt_sdf_hint[i]) if minibatch < 1000 else 0.
                
            loss = loss_latent + loss_random + loss_pc + loss_hint
            
            l.append(float(loss))
            llat.append(float(loss_latent))
            lpc.append(float(loss_pc))
            lrd.append(float(loss_random))
            lh.append(float(loss_hint))
            if minibatch%50 == 49:
                plt.figure(figsize=(8,6))
                plt.yscale('log')
                plt.plot(lpc, label = f'Point cloud loss ({lpc[-1]})')
                plt.plot(lrd, label = f'Other points loss ({lrd[-1]})')
                plt.plot(lh, label = f'Hint points loss ({lh[-1]})')
                plt.plot(llat, label = f'Latent code loss ({llat[-1]})')
                plt.legend()
                plt.show()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
def optimize_latent(net, pc, nc, batch_size, pc_batch_size, epochs, num_hints):
    
    latent = torch.zeros((net.dim_latent), device=device)
    latent = nn.Parameter(latent)
    optim = torch.optim.Adam(lr=2e-5, params=[latent])
    
    pts_hint = torch.rand((num_hints, 3), device = device)*2-1
    pts_hint.requires_grad = True
    gt_sdf_hint, gt_grad_hint = dist_to_pc(pts_hint, pc, nc, with_grad=True)
    gt_sdf_hint = gt_sdf_hint.detach()
    gt_grad_hint = gt_grad_hint.detach()
    
    llat, lpc, lrd, lh = [], [], [], []
    
    for batch in range(epochs):
        
        loss_latent = .01 * torch.linalg.norm(latent, dim=0)
        
        pts_random = torch.rand((batch_size, 3), device = device)*2-1
        pts_random.requires_grad = True
        
        if pc.shape[0] <= pc_batch_size:
            pc_sample = pc
            nc_sample = nc
        else:
            sample = torch.randint(0,pc.shape[0],(pc_batch_size,))
            pc_sample = pc[sample]
            nc_sample = nc[sample]
        
        sdf_random = net(torch.concat((pts_random, latent.repeat((batch_size,1))), dim=1))
        sdf_pc = net(torch.concat((pc_sample, latent.repeat((pc_sample.shape[0],1))), dim=1))
        sdf_hint = net(torch.concat((pts_hint, latent.repeat((num_hints,1))), dim=1))
        
        grad_random = gradient(sdf_random, pts_random)
        loss_random = sdf_loss_others(sdf_random, grad_random)
        loss_pc = sdf_loss_point_cloud(sdf_pc, pc_sample, nc_sample)
        loss_hint = sdf_loss_hints_old(sdf_hint, pts_hint, gt_sdf_hint) if batch < 500 else 0.
        
        loss = loss_latent + loss_random + loss_pc + loss_hint
        
        llat.append(float(loss_latent))
        lpc.append(float(loss_pc))
        lrd.append(float(loss_random))
        lh.append(float(loss_hint))
        
        if batch%50 == 49:
            plt.figure(figsize=(8,6))
            plt.yscale('log')
            plt.plot(lpc, label = f'Point cloud loss ({lpc[-1]})')
            plt.plot(lrd, label = f'Other points loss ({lrd[-1]})')
            plt.plot(lh, label = f'Hint points loss ({lh[-1]})')
            plt.plot(llat, label = f'Latent code loss ({llat[-1]})')
            plt.legend()
            plt.show()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    return latent

s = "../Datasets/data/pointclouds/03467517/"
s = "Objects/"
# dirs = [s+dr for dr in os.listdir(s) if os.path.isdir(s+dr)]
# files = [[dr+"/"+f for f in os.listdir(dr)] for dr in dirs]
# dataset = []
# for f in files:
#     dataset += f
# dataset = [s+dr for dr in os.listdir(s)]
# dataset = random.choices(dataset, k=10)

# dataset = [s+"1ae3b398cea3823b49c212147ab9c105.ply", s+"3c125ee606b03cd263ae8c3a62777578.ply"]
names = ["spot",   "bimba",  "bunny", "bitore",    "hand2",
         "helice", "dragon", "dino",  "pillowbox", "hand",
         "pipe", "protein"]
dataset = [s+n+".obj" for n in names]

# net = ShapeSpaceNet(dim_in=3, dim_hidden=128, dim_out=1, num_layers=8,
#                     num_shapes=10, dim_latent=128).to(device)

# optim = torch.optim.Adam(lr=2e-5, params=net.parameters())

# try:
#     optimize_ss_neural_sdf(net, optim, dataset,
#                         dataset_batch_size=10, dataset_epochs=1,
#                         batch_size=10000, epochs=2000,
#                         num_hints=5000, batch_pc_maxsize=10000)
# except KeyboardInterrupt:
#     pass

# torch.save(net, "Shape Space Networks/ssnn_all10_lat128.net")

net = torch.load("Shape Space Networks/ssnn_all12_lat128_hidden128_TV.net")

# for i,z in enumerate(np.linspace(0,1,1)):
#     for k in range(len(dataset)):
#         display_sdfColor(lambda x:net(torch.concat((x, torch.zeros((x.shape[0],1), device=device)+z, net.latent[k,:].repeat((x.shape[0],1))), dim=1)), 250)

for k in range(len(dataset)):
    f = lambda x:net(torch.concat((x,net.latent[k,:].repeat((x.shape[0],1))), dim=1))
    img = sphere_tracing_gpu(f, 200, 0*np.pi/6, 8*np.pi/6, 1., 3., 1., .02, np.array([-.6,.8,.3]))
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(np.rot90(img))

# pca = PCA(n_components=2)
# pca2 = pca.fit_transform(net.latent.cpu().detach())
# plt.plot(pca2[:,0], pca2[:,1], '.')
# for i in range(len(dataset)):
#     plt.text(pca2[i,0], pca2[i,1], names[i])
# plt.show()

# for t in np.linspace(0,1,10):
#     code = t * net.latent[0,:] + (1-t) * net.latent[7,:]
#     display_sdfColor(lambda x:net(torch.concat((x, torch.zeros((x.shape[0],1), device=device), code.repeat((x.shape[0],1))), dim=1)), 250)
#     display_grad(lambda x:net(torch.concat((x, code.repeat((x.shape[0],1))), dim=1)), 250, 0, 'z')

# for t in np.linspace(0,1,10):
#     code = t * net.latent[4,:] + (1-t) * net.latent[7,:]
#     f = lambda x:net(torch.concat((x,code.repeat((x.shape[0],1))), dim=1))
#     img = sphere_tracing_gpu(f, 200, 0*np.pi/6, 0*np.pi/6, 1., 3., 1., .02, np.array([.6,.8,.3]))
#     plt.figure(figsize=(10,10))
#     plt.axis('off')
#     plt.imshow(np.rot90(img))

# pc,nc,_ = load_mesh(dataset[7])
# pc = -pc
# nc = -nc
# latent = optimize_latent(net, pc, nc, 10000, 10000, 500, 5000)
# latent = torch.rand((128), device=device)
# latent = nn.functional.normalize(latent, dim=0)*0.
# latent = latent[None,:]
# latent = net.latent[2,:]+0.1*nn.functional.normalize(torch.rand((128), device=device), dim=0)
# f = lambda x:net(torch.concat((x,latent.repeat((x.shape[0],1))), dim=1))
# img = sphere_tracing_gpu(f, 200, 0*np.pi/6, 0*np.pi/6, 1., 3., 1., .02, np.array([.6,.8,.3]))
# plt.figure(figsize=(10,10))
# plt.axis('off')
# plt.imshow(np.rot90(img))