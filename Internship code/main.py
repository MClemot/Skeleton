# -*- coding: utf-8 -*-

import torch
from matplotlib import pyplot as plt
import numpy as np

from nn import pretrain, neural_sdf
from skeleton import find_skeleton_gpu, skpoints_to_mesh_gudhi, find_skeleton_elasticity
from geometry import sample_mesh, to_obj
from render import sphere_tracing_gpu
from display import display_sdfColor, display_grad, display_gradgrad
from tools import gauss_curvature, area

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# pretraining
# =============================================================================

net = pretrain(dim_hidden=64, num_layers=8, skip=[], lr=2e-5, batch_size=25000, epochs=5000)
torch.save(net, "Pretrained/pretrained_{}_{}.net".format(64, 8))


# =============================================================================
# neural SDF optimization
# =============================================================================

s = "hand2"
pc, nc = sample_mesh("Objects/{}.obj".format(s), 100000)

net = torch.load("Pretrained/pretrained_{}_{}.net".format(64, 8))

optim = torch.optim.Adam(lr=2e-5, params=net.parameters())

net = neural_sdf(pc, nc,
                  dim_hidden=64, num_layers=8, skip=[],
                  lr=2e-5, batch_size=25000, epochs=1000, pc_batch_size=25000,
                  num_hints=10000, tv_ends=2000, hints_ends=2000, viscosity=False,
                  net = net)

torch.save(net, "Networks/net_{}.net".format(s))
# net = torch.load("Networks/net_{}.net".format(s))


# =============================================================================
# draw some slices
# =============================================================================

for i,z in enumerate(np.linspace(-1,1,10)):
    display_sdfColor(net, 250, z, 'y')
    display_grad(net, 250, z, 'y')
    display_gradgrad(net, 100, z, 'y') 


# =============================================================================
# sphere-tracing rendering
# =============================================================================

img = sphere_tracing_gpu(net, 250, 0*np.pi/6, 1*np.pi/6, 1., 3., 1., .02, np.array([1,1,1]))
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(np.rot90(img))


# =============================================================================
# skeletal points extraction (line-search)
# =============================================================================

sk = find_skeleton_gpu(net, -1, 10000, 50, .5, 1, 100, s, resampling = False)


# =============================================================================
# skeletal points extraction (line-search)
# =============================================================================

sk_energy = find_skeleton_elasticity(net, 250, 1e3, 3e0, 0e3, 0.02)


# =============================================================================
# mesh from skeletal points
# =============================================================================

points, triangles, bones = skpoints_to_mesh_gudhi(sk, 0.03, 0.09)

to_obj(points, "skeleton_{}.obj".format(s), tri=triangles, lines=bones)


# =============================================================================
# topological properties
# =============================================================================

a = area(net, 0.0005)
print("aera: ", a)
K = 0
N = 10
for i in range(N):
    pc, nc = sample_mesh("Objects/{}.obj".format(s), 10000)
    K += gauss_curvature(net, pc).mean().item()/N/(2*np.pi)*a
print("euler characteristic: ", K)
print("genus: ", (2-K)/2)
