# Implementation of `Denoising Diffusion Probabilistic Models` for MNIST dataset

Usage:
Train model using
```bash
$ python train.py
```

Additional argument available.
```bash
$ python train.py --help
usage: train.py [-h] [-t STEPS] [-e EPOCHS] [-b BATCH_SIZE] [-l LEARNING_RATE] [-o OUT_DIR]

Train Diffusion models with DDPM

optional arguments:
  -h, --help            show this help message and exit
  -t STEPS, --steps STEPS
                        Number of timesteps in sampling process $T$ (default = 1000).
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train the model (default = 10).
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for training (default = 64).
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate for optimizer (default = 2e-5)
  -o OUT_DIR, --out-dir OUT_DIR
                        Output directory to save trained model (default = zoo/<timestamp>
```

Sample from trained model
```bash
$ python sample.py -d <trained model directory>
```

Additional arguments available
```bash
$ python sample.py --help
usage: sample.py [-h] [-t STEPS] -d MODEL_DIR

Sample from trained model

optional arguments:
  -h, --help            show this help message and exit
  -t STEPS, --steps STEPS
                        Number of timesteps in sampling process $T$ (default = 1000).
  -d MODEL_DIR, --model-dir MODEL_DIR
                        Directory of saved trained model
```

## References
1. https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html
2. https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/ddpm
