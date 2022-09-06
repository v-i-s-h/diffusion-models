#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --output=logs/job-%A.out
#SBATCH --error=logs/job-%A.err

module load pytorch

python train.py # with default