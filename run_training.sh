#!/bin/bash
#SBATCH --job-name=oko_ood
#SBATCH --partition=gpu-5h
#SBATCH --cpus-per-task=8
#SBATCH --exclude=head024,head021,head025,head047,head076,head074,head050,head075
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%j.out

apptainer run --nv ../oko-ood/oko-ood.sif python train_ssl_cluster.py "$@"