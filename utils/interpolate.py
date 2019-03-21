import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(".")

from utils.anno import gen_images
from utils.vis import imsave_inp
from models.simple_vae import VAE

parser = argparse.ArgumentParser(description='vae.pytorch')
parser.add_argument('--logdir', type=str, default="./log/vae-123")
parser.add_argument('--num', type=int, default=10)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--model', type=str, default="vae-123", choices=["vae-123", "vae-345", "pvae"])
parser.add_argument('--path', type=str, default="./log/vae-123/final_model.pth")
parser.add_argument('--attr', type=str, default="Smiling")
args = parser.parse_args()

# Set GPU (Single GPU usage is only supported so far)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Log
logdir = os.path.join(args.logdir, "interpolate")
if not os.path.exists(logdir):
    os.makedirs(logdir)
# Model
model = VAE(device=device, is_train=False).to(device)
try:
    model.load_state_dict(torch.load(args.path))
except:
    print("Invalid weight path.")
# Loader
gen = gen_images(args.attr, args.num)

with torch.no_grad():
    for i, (p, n) in enumerate(gen):
        p = torch.unsqueeze(p.to(device), 0)
        n = torch.unsqueeze(n.to(device), 0)

        p_z = model.encode(p)
        n_z = model.encode(n)

        # Interpolate between p_z and n_z
        rec_tensors = torch.zeros((11, 3, 64, 64))

        for j, alpha in enumerate(np.arange(0.0, 1.1, 0.1)):
            z_inp = alpha * (p_z - n_z) + n_z
            rec_tensors[j, :, :, :] = model.decode(z_inp)

        imsave_inp(rec_tensors, os.path.join(logdir, f"{args.attr}-{i+1}.png"))
