#!/bin/bash
#SBATCH --gres=gpu:1 
#SBATCH --nodes 1
#SBATCH --time 0:30:00
#SBATCH --mem-per-cpu 10G
#SBATCH --job-name train_gan
#SBATCH --output train_gan.out

python -u train_cgan.py polygon
