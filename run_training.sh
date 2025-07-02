#!/bin/bash
#SBATCH --job-name=oko_ood
#SBATCH --partition=gpu-5h
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out

apptainer run --nv ../oko-ood/oko-ood.sif python train_ssl_cluster.py "$@"