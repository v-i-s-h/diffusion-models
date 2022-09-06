# Implementation of DDPM training algorithm

import os
import argparse
from datetime import datetime
from tokenize import String

import numpy as np
import torch
import torchvision

from tqdm import tqdm
import matplotlib.pyplot as plt

from unet import UNet
from ddpm import DenoiseDiffusion


class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor()
        ])

        super().__init__(root="data/", train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


def main():
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser(description="Train Diffusion models with DDPM")
    parser.add_argument('-t', '--steps', default=1000, type=int,
                        help="Number of timesteps in sampling process $T$ (default = 1000).")
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help="Number of epochs to train the model (default = 10).")
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help="Batch size for training (default = 64).")
    parser.add_argument('-l', '--learning-rate', default=2e-5, type=float,
                        help="Learning rate for optimizer (default = 2e-5)")
    parser.add_argument('-o', '--out-dir', default=os.path.join('zoo', timestamp),
                        help="Output directory to save trained model (default = zoo/<timestamp>")
    args = parser.parse_args()

    # Configs
    n_steps: int = args.steps
    n_epochs: int = args.epochs
    batch_size: int = args.batch_size
    learning_rate: float = args.learning_rate
    outdir: String = args.out_dir
    
    # For MNIST
    image_size = 32 #
    image_channels = 1
    dataset = MNISTDataset(image_size=image_size)

    # Prepare output dirs
    os.makedirs(outdir)
    samples_dir = os.path.join(outdir, 'samples')
    os.makedirs(samples_dir)
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    eps_model = UNet(
        image_channels=image_channels,
        n_channels=64,
        ch_mults=[1, 2, 2, 4],
        is_attn=[False, False, False, True]
    ).to(device=device)

    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=n_steps,
        device=device
    )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.Adam(eps_model.parameters(), lr = learning_rate)

    # noise to be used for sampling images to save - fixed across epochs
    n_samples = 16 # 
    random_noise = torch.randn([n_samples, image_channels, image_size, image_size], device=device)

    # Run training
    for i in range(n_epochs):
        # ------------------------------ Train ---------------------------------
        _loss = []
        for data in tqdm(data_loader, desc='Epoch#{:02d}'.format(i)):
            data = data.to(device)

            optimizer.zero_grad()
            loss = diffusion.loss(data)
            loss.backward()
            optimizer.step()

            _loss.append(loss.item())
        
        # ----------------------------- Sample ---------------------------------
        with torch.no_grad():
            x = random_noise

            for _t in tqdm(range(n_steps), desc="Sampling from model..."):
                t = n_steps - 1 - _t
                x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))

            # Change rows and cols if n_samples change
            n_rows = 4
            n_cols = 4
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows))
            for _i in range(n_rows):
                for _j in range(n_cols):
                    ax[_i, _j].imshow(x[_i*n_cols + _j].cpu().numpy()[0])
                    ax[_i, _j].axis('off')
            plt.savefig(os.path.join(samples_dir, '{:02d}'.format(i)), bbox_inches='tight')

        print("End of epoch {}      Loss = {:.3f}".format(i+1, np.mean(_loss)))

    # Save model
    torch.save(eps_model.state_dict(), os.path.join(outdir, 'model.pth'))
    

if __name__ == "__main__":
    main()
