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
    'batch_size': 64,
    'z_dim': 100,
    'lr': 1e-4,
    'n_epoch': 50,
    'k_epoch': 1,
    'data_dir': sys.argv[1],
    'log_dir': 'log_sngan',
    'save_path': sys.argv[2],
    'device': 'cuda'
}
args = argparse.Namespace(**args)
os.makedirs(args.log_dir, exist_ok=True)

# set random seed
same_seeds(0)

# dataset
dataset = get_dataset(args.data_dir)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

# model
G = Generator_sn(args.z_dim).to(args.device)
D = Discriminator_sn(3).to(args.device)

# loss
criterion = nn.BCELoss()

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

""" training """
z_sample = Variable(torch.randn(100, args.z_dim), requires_grad=False).to(args.device)
loss_G = -1

for epoch in range(args.n_epoch):
    for i, x_real in enumerate(dataloader):
        x_real = x_real.to(args.device)
        batch_size = x_real.size(0)
        label_real = torch.ones((batch_size)).to(args.device)
        label_fake = torch.zeros((batch_size)).to(args.device)
        # train D
        z_fake = Variable(torch.randn(batch_size, args.z_dim), requires_grad=False).to(args.device)
        x_fake = G(z_fake)
        
        logit_real = D(x_real)
        logit_fake = D(x_fake)

        loss_D = (criterion(logit_real, label_real) + criterion(logit_fake, label_fake)) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()
        
        # train G every k_epoch
        if (i + 1) % args.k_epoch == 0:
            z_fake = Variable(torch.randn(batch_size, args.z_dim), requires_grad=False).to(args.device)
            x_fake = G(z_fake)

            logit_fake = D(x_fake)
            loss_G = criterion(logit_fake, label_real)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print('Epoch {}/{} Step {}/{} loss_D {:.5f} loss_G {:.5f}'.format(
                epoch + 1, args.n_epoch,
                i + 1, len(dataloader),
                loss_D, loss_G
            ), end='\r')
    print('')
    
    # show generate figure
    x_sample = (G(z_sample).data + 1) / 2.0
    grid_image = torchvision.utils.make_grid(x_sample.cpu(), nrow=10)

    filename = os.path.join(args.log_dir, f'Epoch_{epoch+1:03d}.jpg')
    torchvision.utils.save_image(grid_image, filename, nrow=10)

    #plt.figure(figsize=(10, 10))
    #plt.imshow(grid_image.permute(1, 2, 0))
    #plt.show()
    
    # save D & G
    torch.save(G.state_dict(), args.save_path)
