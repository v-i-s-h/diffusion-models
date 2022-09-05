from random import shuffle
import numpy as np
import torch
import torchvision

from tqdm import tqdm

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

    # Configs
    n_steps: int = 1000
    n_epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 2e-5
    image_size = 32

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    eps_model = UNet(
        image_channels=1,
        n_channels=64,
        ch_mults=[1, 2, 2, 4],
        is_attn=[False, False, False, True]
    ).to(device=device)

    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=n_steps,
        device=device
    )

    dataset = MNISTDataset(image_size=image_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.Adam(eps_model.parameters(), lr = learning_rate)

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
            print("    loss = {:.4f}".format(_loss[-1]))

        # ----------------------------- Sample ---------------------------------

        print("End of epoch {}      Loss = {:.3f}".format(i+1, np.mean(_loss)))

    

if __name__ == "__main__":
    main()
