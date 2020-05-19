## import os
import sys
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable
from dataset import *
from model import *
from utils import *

""" setup """
# hyperparameters
args = {
    'z_dim': 100,
    'save_path': sys.argv[1],
    'img_path': sys.argv[2],
    'img_row': 10,
    'img_col': 5,
    'device': 'cuda'
}
args = argparse.Namespace(**args)

# set random seed
same_seeds(0)

# model
G = Generator(args.z_dim).to(args.device)
G.load_state_dict(torch.load(args.save_path))

# generate
G.eval()
z_sample = Variable(torch.randn(args.img_row * args.img_col, args.z_dim), requires_grad=False).to(args.device)
x_sample = (G(z_sample).data + 1) / 2.0
torchvision.utils.save_image(x_sample, args.img_path, nrow=args.img_row)
