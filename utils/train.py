import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.append(".")

from utils.data import get_celeba_loaders
from utils.vis import plot_loss, imsave, Logger
from models.simple_vae import VAE, FLPLoss, KLDLoss

parser = argparse.ArgumentParser(description='vae.pytorch')
parser.add_argument('--logdir', type=str, default="./log/celeba")
parser.add_argument('--batch_train', type=int, default=256)
parser.add_argument('--batch_test', type=int, default=16)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--initial_lr', type=float, default=0.0005)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.5)

args = parser.parse_args()

# Set GPU (Single GPU usage is only supported so far)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Dataloader
dataloaders = get_celeba_loaders(args.batch_train, args.batch_test)
# Model
model = VAE(device=device).to(device)
# Loss
flp_criterion = FLPLoss(device)
kld_criterion = KLDLoss(device)
# Solver
optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
# Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
# Log
logdir = args.logdir
if not os.path.exists(logdir):
    os.makedirs(logdir)
# Logger
logger = Logger(os.path.join(logdir, "log.txt"))

# History
history = {"train": [], "test": []}

# Start training
for epoch in range(args.epochs):
    for phase in ["train", "test"]:
        if phase == "train":
            model.train(True)
            logger.write(f"\n----- Epoch {epoch+1} -----")
        else:
            model.train(False)

        # Loss
        running_loss = 0.0
        # Data num
        data_num = 0

        # Train
        for i, x in enumerate(dataloaders[phase]):
            # Optimize params
            if phase == "train":
                optimizer.zero_grad()

                # Pass forward
                x = x.to(device)
                latent_z, rec_x = model(x)

                # Calc loss
                flp_loss = flp_criterion(x, rec_x)
                kld_loss = kld_criterion(latent_z)
                loss = args.alpha * kld_loss + args.beta * flp_loss

                loss.backward()
                optimizer.step()
    
            elif phase == "test":
                with torch.no_grad():
                    optimizer.zero_grad()

                    # Pass forward
                    x = x.to(device)
                    latent_z, rec_x = model(x)

                    # Calc loss
                    flp_loss = flp_criterion(x, rec_x)
                    kld_loss = kld_criterion(latent_z)
                    loss = args.alpha * kld_loss + args.beta * flp_loss

            # Add stats
            running_loss += loss * x.size(0)
            data_num += x.size(0)

            # Visualize
            if i % 100 == 0:
                imsave(x, rec_x, os.path.join(logdir, f"epoch{epoch+1}", phase, f"batch{i}.png"), 4, 4)

        # Log
        epoch_loss = running_loss / data_num
        logger.write(f"{phase} Loss : {epoch_loss:.4f}")
        history[phase].append(epoch_loss)

        if phase == "test":
            plot_loss(logdir, history)