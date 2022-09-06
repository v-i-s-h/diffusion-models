# Sample from a trained model

import os
import argparse

import torch

from matplotlib import pyplot as plt
from tqdm import tqdm

from unet import UNet
from ddpm import DenoiseDiffusion

def main():
    parser = argparse.ArgumentParser(description="Sample from trained model")
    parser.add_argument('-t', '--steps', default=1000, type=int,
                        help="Number of timesteps in sampling process $T$ (default = 1000).")
    parser.add_argument('-d', '--model-dir', required=True,
                        help="Directory of saved trained model")
    args = parser.parse_args()

    n_steps: int = args.steps
    model_dir: str = args.model_dir

    image_size = 32 #
    image_channels = 1

    # To save generated animation
    samples_dir = os.path.join(model_dir, 'generated')
    os.makedirs(samples_dir, exist_ok=True) # Overwrite if exists

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Load model
    model = UNet(image_channels, 
                n_channels=64, 
                ch_mults=[1, 2, 2, 4], 
                is_attn=[False, False, False, True])
    model.load_state_dict(torch.load(
                            os.path.join(model_dir, 'model.pth'), 
                            map_location=device))
    model = model.to(device=device)

    diffusion = DenoiseDiffusion(
        eps_model=model,
        n_steps=n_steps,
        device=device)

    print(">>", next(model.parameters()).is_cuda)

    # noise to be used for sampling images to save - fixed across epochs
    n_samples = 25 # 
    random_noise = torch.randn([n_samples, image_channels, image_size, image_size], device=device)  

    with torch.no_grad():
        x = random_noise
        print(">>>>>>", x.is_cuda)

        for _t in tqdm(range(n_steps), desc="Sampling from model..."):
            t = n_steps - 1 - _t
            x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))

            if (_t % 10 == 0) or (_t == (n_steps - 1)):
                # Change rows and cols if n_samples change
                n_rows = 5
                n_cols = 5
                fig, ax = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows))
                for _i in range(n_rows):
                    for _j in range(n_cols):
                        ax[_i, _j].imshow(x[_i*n_cols + _j].cpu().numpy()[0])
                        ax[_i, _j].axis('off')
                plt.suptitle('t = {}/{}'.format(_t, n_steps))
                plt.savefig(os.path.join(samples_dir, '{:04d}'.format(_t)), bbox_inches='tight')
                plt.close()

if __name__ == '__main__':
    main()
