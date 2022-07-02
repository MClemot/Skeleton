# Usage

usage: main.py [-h] [-pretrain PRE PRE] [-i INPUT] [-mshsmp MSHSMP] [-prenet PRENET] [-lr LR] [-epochs EPOCHS] [-bs BS] [-numlpts NUMLPTS] [-cutlpts CUTLPTS] [-nsk NSK] [-steps STEPS] [-radius RADIUS] [-alpha ALPHA] -o OUTPUT

optional arguments:
  -h, --help         show this help message and exit
  
  -pretrain PRE PRE  pretrain network with {arg1} hidden layers with {arg0} neurons
  
  -i INPUT           input file

  -mshsmp MSHSMP     number of points sampled onto the mesh
  
  -prenet PRENET     pretrained net file
  
  -lr LR             learning rate
  
  -epochs EPOCHS     epochs
  
  -bs BS             batch size
  
  -numlpts NUMLPTS   number of learning points

  -cutlpts CUTLPTS   which iteration the learning points loss is cut

  -nsk NSK           number of sampled skeleton points
  
  -steps STEPS       number of time the line search along the gradient direction is done
  
  -radius RADIUS     radius of the ball where other skeletal points are deleted
  
  -alpha ALPHA       alpha value for alpha-complex computation
  
  -o OUTPUT          output file
  
# Reproducing the experiments

## Pretraining a network

python main.py -pretrain 128 8 -epochs 2500 -o pretrained_128_8.net

## Skeletonizing a sample of points from a mesh (.obj)

python main.py -i Objects\metatron.obj -prenet pretrained_128_8.net -steps 2 -nsk 20000 -o sk_metatron.obj

python main.py -i Objects\hilbert.obj -prenet pretrained_128_8.net -steps 2 -radius .05 -alpha .1 -o sk_hilbert.obj

## Skeletonizing from a point cloud with normals (.ply without header i.e. with lines of the form 'x y z nx ny nz \n')

python main.py -i Objects\guitar3.ply -prenet pretrained_128_8.net -o sk_guitar.obj
