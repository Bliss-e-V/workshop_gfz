# SLURM Cluster Training for SSL Models

This repository contains scripts for training self-supervised learning models on a SLURM cluster using E-OBS precipitation data.

## Files

- **`train_ssl_cluster.py`**: Main training script optimized for cluster environment
- **`train_ssl.slurm`**: SLURM job script for submitting training jobs
- **`SLURM_TRAINING_README.md`**: This documentation file

## Models Trained

### 1. Temporal Prediction Model
- **Task**: Predict next day's precipitation from past 7 days
- **Architecture**: ConvLSTM with 4 layers, 128 hidden channels
- **Model size**: ~16M parameters (full capacity)
- **Training**: 50 epochs with early stopping

### 2. Masked Modeling Model  
- **Task**: Reconstruct masked spatial regions of precipitation maps
- **Architecture**: U-Net with 5 levels, 128 base channels
- **Model size**: ~45M parameters (full capacity)
- **Training**: 40 epochs with early stopping

## Quick Start

### 1. Submit Both Models (Recommended)
```bash
sbatch train_ssl.slurm
```

### 2. Train Individual Models
```bash
# Only temporal prediction
python train_ssl_cluster.py --model temporal --epochs 50

# Only masked modeling
python train_ssl_cluster.py --model masked --epochs 40

# Both with different epoch counts
python train_ssl_cluster.py --model both --temporal-epochs 60 --masked-epochs 50
```

## Configuration

### Model Configurations (Optimized for Performance)
```python
# Temporal Prediction Model
TEMPORAL_CONFIG = {
    'hidden_channels': 128,     # Full capacity (vs 32 for Mac)
    'num_layers': 4,            # Full depth (vs 2 for Mac)
    'spatial_size': (128, 128)  # Higher resolution (vs 64x64)
}

# Masked Modeling Model
MASKED_CONFIG = {
    'base_channels': 128,       # Full capacity (vs 32 for Mac)
    'num_levels': 5,            # Full depth (vs 3 for Mac)
}
```

### Training Configurations
```python
TRAINING_CONFIG = {
    'batch_size': 64,           # Large batch for cluster
    'num_workers': 8,           # Multi-threaded loading
    'train_samples': 20000,     # Full dataset
    'val_samples': 4000,
    'early_stopping_patience': 15  # Patience for long training
}
```

## Output Structure

### Model Saves Location: `/home/pinetzki/hydro_models/`

```
/home/pinetzki/hydro_models/
├── temporal_prediction_final.pth           # Final temporal model weights
├── temporal_prediction_config.json         # Temporal model configuration
├── temporal_prediction_checkpoints/        # Training checkpoints
│   ├── temporal_prediction_epoch=XX_val_loss=X.XXXX.ckpt
│   └── last.ckpt
├── masked_modeling_final.pth               # Final masked model weights  
├── masked_modeling_config.json             # Masked model configuration
├── masked_modeling_checkpoints/            # Training checkpoints
│   ├── masked_modeling_epoch=XX_val_loss=X.XXXX.ckpt
│   └── last.ckpt
└── training_results.json                   # Final training summary
```

### Logs Location: `logs/`

```
logs/
├── ssl_training_JOBID.out                  # SLURM stdout
├── ssl_training_JOBID.err                  # SLURM stderr  
├── temporal_prediction/                    # TensorBoard logs
└── masked_modeling/                        # TensorBoard logs
```

## Expected Training Times

- **Temporal Model**: ~3-4 hours for 50 epochs
- **Masked Model**: ~4-5 hours for 40 epochs  
- **Both Models**: ~7-9 hours total

## Memory Usage

- **GPU Memory**: ~12-16GB (requires V100/A100 class GPU)
- **System RAM**: ~20-25GB
- **Disk Space**: ~5-10GB for checkpoints and logs

## Customization

### Adjust SLURM Parameters
Edit `train_ssl.slurm`:
```bash
#SBATCH --mem=32G           # Increase if needed
#SBATCH --time=12:00:00     # Adjust time limit
#SBATCH --gres=gpu:1        # GPU requirements
```

### Modify Training Parameters
Edit `train_ssl_cluster.py`:
```python
# In ClusterConfig class
'batch_size': 64,          # Reduce if GPU memory issues
'train_samples': 20000,    # Reduce for faster training
'learning_rate': 1e-3,     # Adjust learning rate
```

## Monitoring Training

### Check Job Status
```bash
squeue -u pinetzki
```

### Monitor GPU Usage
```bash
ssh node_name
nvidia-smi -l 1
```

### View Live Logs
```bash
tail -f logs/ssl_training_JOBID.out
```

### TensorBoard (if available)
```bash
tensorboard --logdir=logs --port=6006
```

## Expected Results

### Temporal Prediction Model
- **Validation Loss**: ~0.15-0.25 (MSE on normalized data)
- **Performance**: Good at capturing seasonal patterns and trends
- **Use Cases**: Weather forecasting, drought prediction

### Masked Modeling Model  
- **Validation Loss**: ~0.10-0.20 (reconstruction loss)
- **Performance**: Good at spatial pattern completion
- **Use Cases**: Gap filling, spatial interpolation, data quality control

## Troubleshooting

### Common Issues

1. **GPU Out of Memory**:
   - Reduce `batch_size` from 64 to 32 or 16
   - Reduce `spatial_size` from (128,128) to (64,64)

2. **Training Too Slow**:
   - Reduce `train_samples` for faster iterations
   - Use fewer epochs for initial testing

3. **NaN Losses**:
   - The code includes NaN handling, but check data quality
   - Consider reducing learning rate

4. **Module Not Found**:
   - Adjust module loading in SLURM script
   - Check conda environment setup

### Getting Help

Check the training logs and error messages:
```bash
# View stdout
cat logs/ssl_training_JOBID.out

# View stderr  
cat logs/ssl_training_JOBID.err

# View training results
cat /home/pinetzki/hydro_models/training_results.json
```

## Advanced Usage

### Resume from Checkpoint
The training automatically saves checkpoints. To resume:
```python
# In the training script, modify trainer creation:
trainer = L.Trainer(
    resume_from_checkpoint="/path/to/checkpoint.ckpt",
    # ... other parameters
)
```

### Multi-GPU Training
For multiple GPUs, modify the trainer:
```python
trainer = L.Trainer(
    devices=2,              # Use 2 GPUs
    strategy="ddp",         # Distributed training
    # ... other parameters
)
```

## Citation

If you use these models in research, please cite the relevant papers and datasets:
- E-OBS dataset: Cornes et al. (2018)
- PyTorch Lightning: Falcon et al. (2019) 