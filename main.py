# -*- coding: utf-8 -*-

import torch
import argparse

from nn import pretrain, optimize_neural_sdf
from skeleton import find_skeleton_gpu, skpoints_to_mesh_gudhi, reduce
from geometry import load_mesh, to_obj, from_obj, sample_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser()


parser.add_argument('-pretrain',
                    dest='pre',
                    help='pretrain network with {arg1} hidden layers with {arg0} neurons',
                    type=int,
                    nargs=2
                    )

parser.add_argument('-i',
                    dest='input',
                    help='input file'
                    )

parser.add_argument('-prenet',
                    dest='prenet',
                    help='pretrained net file'
                    )

parser.add_argument('-lr',
                    dest='lr',
                    default=2e-5,
                    help='learning rate',
                    type=float
                    )

parser.add_argument('-epochs',
                    dest='epochs',
                    default=1000,
                    help='epochs',
                    type=int
                    )

parser.add_argument('-bs',
                    dest='bs',
                    default=25000,
                    help='batch size',
                    type=int
                    )

parser.add_argument('-numlpts',
                    dest='numlpts',
                    default=5000,
                    help='number of learning points',
                    type=int
                    )

parser.add_argument('-steps',
                    dest='steps',
                    default=1,
                    help='number of time the line search along the gradient direction is done',
                    type=int
                    )

parser.add_argument('-radius',
                    dest='radius',
                    default=0.03,
                    help='radius of the ball where other skeletal points are deleted',
                    type=float
                    )

parser.add_argument('-alpha',
                    dest='alpha',
                    default=0.06,
                    help='alpha value for alpha-complex computation',
                    type=float
                    )

parser.add_argument('-o',
                    required=True,
                    dest='output',
                    help='output file'
                    )

args = parser.parse_args()

if args.pre != None:
    net = pretrain(dim_hidden=args.pre[0], num_layers=args.pre[1], skip=[], lr=args.lr, batch_size=25000, epochs=args.epochs)
    torch.save(net, args.output)

if args.input != None:
    if args.input[-3:] == 'net':
        net = torch.load(args.input)
    else:
        pc, nc = sample_mesh(args.input, 100000)
        
        net = torch.load(args.prenet)
        
        optim = torch.optim.Adam(lr=args.lr, params=net.parameters())
        
        try:
            optimize_neural_sdf(net, optim, pc, nc,
                                batch_size=args.bs, pc_batch_size=args.bs, epochs=args.epochs,
                                nb_hints=args.numlpts, tv_ends=args.epochs, hints_ends=args.epochs)
        except KeyboardInterrupt:
            pass
        
        torch.save(net, args.output[:-3]+"net")
    
    sk = find_skeleton_gpu(net, 10000, 50, .5, args.steps, 100, resampling = False)
    
    points, triangles, bones = skpoints_to_mesh_gudhi(sk, args.radius, args.alpha)
    
    to_obj(points, args.output, tri=triangles, lines=bones)