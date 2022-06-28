# SkeletonLearning
Learning a skeleton from a shape

Internship from Mattéo Clémot (ENS Lyon) February-July 2022.

## Code

# Usage

usage: main.py [-h] [-pretrain PRE PRE] [-i INPUT] [-prenet PRENET] [-lr LR] [-epochs EPOCHS] [-bs BS] [-numlpts NUMLPTS] [-steps STEPS]
               [-radius RADIUS] [-alpha ALPHA] -o OUTPUT

optional arguments:
  -h, --help         show this help message and exit
  -pretrain PRE PRE  pretrain network with {arg1} hidden layers with {arg0} neurons
  -i INPUT           input file
  -prenet PRENET     pretrained net file
  -lr LR             learning rate
  -epochs EPOCHS     epochs
  -bs BS             batch size
  -numlpts NUMLPTS   number of learning points
  -steps STEPS       number of time the line search along the gradient direction is done
  -radius RADIUS     radius of the ball where other skeletal points are deleted
  -alpha ALPHA       alpha value for alpha-complex computation
  -o OUTPUT          output file