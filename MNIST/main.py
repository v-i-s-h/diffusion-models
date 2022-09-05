from random import shuffle
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

    # Configs
    n_steps: int = 1000
    n_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 2e-5
    image_size = 32
    image_channels = 1
    n_samples = 16 # Samples to sample at end of every epoch for test

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
        
        # ----------------------------- Sample ---------------------------------
        with torch.no_grad():
            x = torch.randn([n_samples, image_channels, image_size, image_size], device=device)

            for _t in tqdm(range(n_steps), desc="Sampling from model..."):
                t = n_steps - 1 - _t
                x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))

            fig, ax = plt.subplots(1, n_samples, figsize=(2.5*n_samples, 2.5))
            for j in range(n_samples):
                ax[j].imshow(x[j].cpu().numpy()[0])
                ax[j].axis('off')
            plt.savefig('zoo/{:02d}'.format(i), bbox_inches='tight')


        print("\nEnd of epoch {}      Loss = {:.3f}".format(i+1, np.mean(_loss)))

    # Save model
    torch.save(eps_model.state_dict(), 'zoo/eps_model.pth')

    

if __name__ == "__main__":
    main()
