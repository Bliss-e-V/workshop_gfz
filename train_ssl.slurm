#!/bin/bash
#SBATCH --job-name=ssl_hydro_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ssl_training_%j.out
#SBATCH --error=logs/ssl_training_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pinetzki@tu-berlin.de

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load modules (adjust based on your cluster)
module load anaconda3
module load cuda/11.8
module load cudnn/8.2.4

# Activate conda environment (adjust based on your setup)
source activate pytorch

# Create necessary directories
mkdir -p logs
mkdir -p /home/pinetzki/hydro_models

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print system information
echo "CUDA Device Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

# Run training script - train both models with extended epochs
echo "Starting SSL training..."
python train_ssl_cluster.py \
    --model both \
    --temporal-epochs 50 \
    --masked-epochs 40

echo "Training completed at: $(date)"

# Print final model information
echo "Saved models:"
ls -la /home/pinetzki/hydro_models/

# Display final results
if [ -f "/home/pinetzki/hydro_models/training_results.json" ]; then
    echo "Training Results:"
    cat /home/pinetzki/hydro_models/training_results.json
fi 