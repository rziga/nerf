#!/bin/bash
#SBATCH --job-name=NeLF
##SBATCH --reservation=fri
#SBATCH --output=output.out
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-10:00:00

srun nvidia-smi
srun python train.py \
    --root /d/hpc/projects/FRI/zr13891/datasets/nerfs/lego/lego \
    --checkpoint ../../checkpoints \
    --train_steps 100000 \
    --batch_size 12