#!/usr/bin/env python3
"""
SLURM Cluster Training Script for Self-Supervised Learning Approaches
Trains both Temporal Prediction and Masked Modeling models on E-OBS data
Includes Weights & Biases (wandb) logging for experiment tracking

Usage:
    python train_ssl_cluster.py --model temporal --epochs 50
    python train_ssl_cluster.py --model masked --epochs 40
    python train_ssl_cluster.py --model both --epochs 50
    python train_ssl_cluster.py --model both --epochs 50 --wandb-project my-project --wandb-entity my-username
"""

import sys
sys.path.append('/home/pinetzki/workshop_gfz/src/')

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

# Add src to path
sys.path.append('src')
from data_utils import EOBSDataLoader, EOBSTemporalPredictionDataset, EOBSMaskedModelingDataset
from models import (
    TemporalPredictionModel, 
    MaskedModelingModel, 
    TemporalPredictionLightningModule,
    MaskedModelingLightningModule
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ClusterConfig:
    """Configuration for cluster training - optimized for performance"""
    
    # Model configurations (full capacity for cluster)
    TEMPORAL_CONFIG = {
        'input_channels': 1,
        'hidden_channels': 128,  # Increased from 32 for better performance
        'num_layers': 4,         # Increased from 2 for better performance
        'sequence_length': 7,
        'prediction_horizon': 1,
        'spatial_size': (128, 128)  # Larger spatial resolution
    }
    
    MASKED_CONFIG = {
        'input_channels': 1,
        'base_channels': 128,    # Increased from 32 for better performance
        'num_levels': 5,         # Increased from 3 for better performance
        'temporal_context': 1
    }
    
    # Training configurations
    TRAINING_CONFIG = {
        'batch_size': 64,        # Larger batch size for cluster
        'num_workers': 8,        # More workers for cluster
        'learning_rate': 1e-3,   # Standard learning rate for longer training
        'weight_decay': 1e-4,
        'max_epochs_temporal': 50,
        'max_epochs_masked': 40,
        'early_stopping_patience': 15,  # More patience for longer training
        'train_samples': 20000,  # Full dataset size
        'val_samples': 4000,
        'precision': 32,
        'gradient_clip_val': 1.0
    }
    
    # Paths
    MODEL_SAVE_DIR = Path("/home/pinetzki/hydro_models")
    DATA_DIR = "src/data"
    LOG_DIR = "logs"
    
    # Wandb configuration
    WANDB_CONFIG = {
        'project': 'hydrology-ssl',
        'entity': None,  # Use default wandb entity
        'tags': ['slurm', 'cluster', 'ssl', 'eobs', 'precipitation'],
        'group': 'cluster_training',
    }


def setup_model_save_dir():
    """Create model save directory if it doesn't exist"""
    ClusterConfig.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model save directory: {ClusterConfig.MODEL_SAVE_DIR}")


def load_eobs_data():
    """Load E-OBS precipitation data"""
    logger.info("Loading E-OBS data...")
    eobs_loader = EOBSDataLoader(data_dir=ClusterConfig.DATA_DIR)
    eobs_data = eobs_loader.load_all_data()
    
    if 'precipitation_mean' in eobs_data:
        precip_data = eobs_data['precipitation_mean']
        logger.info(f"Precipitation data shape: {precip_data.dims}")
        # Get the precipitation variable for min/max calculation
        precip_var = precip_data['rr'] if 'rr' in precip_data.data_vars else list(precip_data.data_vars.values())[0]
        min_val = float(precip_var.min().item())
        max_val = float(precip_var.max().item())
        logger.info(f"Precipitation data range: {min_val:.3f} to {max_val:.3f}")
        return precip_data
    else:
        raise ValueError("No precipitation data found in E-OBS dataset")


def create_temporal_datasets(precip_data):
    """Create temporal prediction datasets"""
    logger.info("Creating temporal prediction datasets...")
    
    train_dataset = EOBSTemporalPredictionDataset(
        precipitation_data=precip_data,
        sequence_length=ClusterConfig.TEMPORAL_CONFIG['sequence_length'],
        prediction_horizon=ClusterConfig.TEMPORAL_CONFIG['prediction_horizon'],
        spatial_crop_size=ClusterConfig.TEMPORAL_CONFIG['spatial_size'],
        normalize=True,
        log_transform=True
    )
    
    val_dataset = EOBSTemporalPredictionDataset(
        precipitation_data=precip_data,
        sequence_length=ClusterConfig.TEMPORAL_CONFIG['sequence_length'],
        prediction_horizon=ClusterConfig.TEMPORAL_CONFIG['prediction_horizon'],
        spatial_crop_size=ClusterConfig.TEMPORAL_CONFIG['spatial_size'],
        normalize=True,
        log_transform=True
    )
    
    # Limit dataset sizes
    train_size = min(len(train_dataset), ClusterConfig.TRAINING_CONFIG['train_samples'])
    val_size = min(len(val_dataset), ClusterConfig.TRAINING_CONFIG['val_samples'])
    
    train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(val_dataset, range(val_size))
    
    logger.info(f"Temporal datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def create_masked_datasets(precip_data):
    """Create masked modeling datasets"""
    logger.info("Creating masked modeling datasets...")
    
    train_dataset = EOBSMaskedModelingDataset(
        precipitation_data=precip_data,
        spatial_size=ClusterConfig.TEMPORAL_CONFIG['spatial_size'],  # Use same spatial size
        mask_ratio=0.25,
        mask_strategy='random_patches',
        patch_size=16,  # Larger patches for more challenging task
        normalize=True,
        log_transform=True,
        temporal_context=ClusterConfig.MASKED_CONFIG['temporal_context']
    )
    
    val_dataset = EOBSMaskedModelingDataset(
        precipitation_data=precip_data,
        spatial_size=ClusterConfig.TEMPORAL_CONFIG['spatial_size'],
        mask_ratio=0.25,
        mask_strategy='random_patches',
        patch_size=16,
        normalize=True,
        log_transform=True,
        temporal_context=ClusterConfig.MASKED_CONFIG['temporal_context']
    )
    
    # Limit dataset sizes
    train_size = min(len(train_dataset), ClusterConfig.TRAINING_CONFIG['train_samples'])
    val_size = min(len(val_dataset), ClusterConfig.TRAINING_CONFIG['val_samples'])
    
    train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(val_dataset, range(val_size))
    
    logger.info(f"Masked datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset):
    """Create data loaders optimized for cluster"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=ClusterConfig.TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=ClusterConfig.TRAINING_CONFIG['num_workers'],
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=ClusterConfig.TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=ClusterConfig.TRAINING_CONFIG['num_workers'],
        persistent_workers=True,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_temporal_model():
    """Create temporal prediction model with full capacity"""
    logger.info("Creating temporal prediction model...")
    
    model = TemporalPredictionModel(**ClusterConfig.TEMPORAL_CONFIG)
    lightning_module = TemporalPredictionLightningModule(
        model=model,
        learning_rate=ClusterConfig.TRAINING_CONFIG['learning_rate'],
        weight_decay=ClusterConfig.TRAINING_CONFIG['weight_decay']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Temporal model - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    return lightning_module


def create_masked_model():
    """Create masked modeling model with full capacity"""
    logger.info("Creating masked modeling model...")
    
    model = MaskedModelingModel(**ClusterConfig.MASKED_CONFIG)
    lightning_module = MaskedModelingLightningModule(
        model=model,
        learning_rate=ClusterConfig.TRAINING_CONFIG['learning_rate'],
        weight_decay=ClusterConfig.TRAINING_CONFIG['weight_decay']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Masked model - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    return lightning_module


def create_callbacks(model_name: str, max_epochs: int):
    """Create training callbacks"""
    callbacks = []
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=ClusterConfig.TRAINING_CONFIG['early_stopping_patience'],
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop)
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=ClusterConfig.MODEL_SAVE_DIR / f"{model_name}_checkpoints",
        filename=f"{model_name}_" + "{epoch:02d}_{val_loss:.4f}",
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def train_temporal_model(precip_data, epochs: int):
    """Train temporal prediction model"""
    logger.info("=" * 60)
    logger.info("TRAINING TEMPORAL PREDICTION MODEL")
    logger.info("=" * 60)
    
    # Create datasets and loaders
    train_dataset, val_dataset = create_temporal_datasets(precip_data)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)
    
    # Create model
    lightning_module = create_temporal_model()
    
    # Create logger
    run_name = f"temporal_cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = WandbLogger(
        name=run_name,
        project=ClusterConfig.WANDB_CONFIG['project'],
        entity=ClusterConfig.WANDB_CONFIG['entity'],
        tags=ClusterConfig.WANDB_CONFIG['tags'] + ['temporal_prediction'],
        group=ClusterConfig.WANDB_CONFIG['group'],
        job_type='temporal_prediction',
        save_dir=ClusterConfig.LOG_DIR,
        log_model='all'  # Log model checkpoints to wandb
    )
    
    # Log hyperparameters
    wandb_logger.experiment.config.update({
        'model_type': 'temporal_prediction',
        'model_config': ClusterConfig.TEMPORAL_CONFIG,
        'training_config': ClusterConfig.TRAINING_CONFIG,
        'epochs': epochs,
        'device': 'gpu' if torch.cuda.is_available() else 'cpu'
    })
    
    # Create callbacks
    callbacks = create_callbacks("temporal_prediction", epochs)
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=ClusterConfig.TRAINING_CONFIG['precision'],
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=ClusterConfig.TRAINING_CONFIG['gradient_clip_val'],
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False  # For speed on cluster
    )
    
    # Train model
    logger.info(f"Starting temporal prediction training for {epochs} epochs...")
    start_time = datetime.now()
    trainer.fit(lightning_module, train_loader, val_loader)
    end_time = datetime.now()
    
    # Log training results
    training_time = end_time - start_time
    logger.info(f"Temporal prediction training completed in {training_time}")
    logger.info(f"Best validation loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
    
    # Save final model
    final_model_path = ClusterConfig.MODEL_SAVE_DIR / "temporal_prediction_final.pth"
    torch.save(lightning_module.model.state_dict(), final_model_path)
    logger.info(f"Final temporal model saved to: {final_model_path}")
    
    # Save training config
    config_path = ClusterConfig.MODEL_SAVE_DIR / "temporal_prediction_config.json"
    config = {
        'model_config': ClusterConfig.TEMPORAL_CONFIG,
        'training_config': ClusterConfig.TRAINING_CONFIG,
        'training_time': str(training_time),
        'epochs_completed': trainer.current_epoch,
        'best_val_loss': float(trainer.callback_metrics.get('val_loss', float('inf')))
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Log final metrics to wandb
    wandb_logger.experiment.log({
        'final/training_time_seconds': training_time.total_seconds(),
        'final/epochs_completed': trainer.current_epoch,
        'final/best_val_loss': float(trainer.callback_metrics.get('val_loss', float('inf')))
    })
    
    return lightning_module, trainer


def train_masked_model(precip_data, epochs: int):
    """Train masked modeling model"""
    logger.info("=" * 60)
    logger.info("TRAINING MASKED MODELING MODEL")
    logger.info("=" * 60)
    
    # Create datasets and loaders
    train_dataset, val_dataset = create_masked_datasets(precip_data)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)
    
    # Create model
    lightning_module = create_masked_model()
    
    # Create logger
    run_name = f"masked_cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = WandbLogger(
        name=run_name,
        project=ClusterConfig.WANDB_CONFIG['project'],
        entity=ClusterConfig.WANDB_CONFIG['entity'],
        tags=ClusterConfig.WANDB_CONFIG['tags'] + ['masked_modeling'],
        group=ClusterConfig.WANDB_CONFIG['group'],
        job_type='masked_modeling',
        save_dir=ClusterConfig.LOG_DIR,
        log_model='all'  # Log model checkpoints to wandb
    )
    
    # Log hyperparameters
    wandb_logger.experiment.config.update({
        'model_type': 'masked_modeling',
        'model_config': ClusterConfig.MASKED_CONFIG,
        'training_config': ClusterConfig.TRAINING_CONFIG,
        'epochs': epochs,
        'device': 'gpu' if torch.cuda.is_available() else 'cpu'
    })
    
    # Create callbacks
    callbacks = create_callbacks("masked_modeling", epochs)
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=ClusterConfig.TRAINING_CONFIG['precision'],
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=ClusterConfig.TRAINING_CONFIG['gradient_clip_val'],
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False  # For speed on cluster
    )
    
    # Train model
    logger.info(f"Starting masked modeling training for {epochs} epochs...")
    start_time = datetime.now()
    trainer.fit(lightning_module, train_loader, val_loader)
    end_time = datetime.now()
    
    # Log training results
    training_time = end_time - start_time
    logger.info(f"Masked modeling training completed in {training_time}")
    logger.info(f"Best validation loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
    
    # Save final model
    final_model_path = ClusterConfig.MODEL_SAVE_DIR / "masked_modeling_final.pth"
    torch.save(lightning_module.model.state_dict(), final_model_path)
    logger.info(f"Final masked model saved to: {final_model_path}")
    
    # Save training config
    config_path = ClusterConfig.MODEL_SAVE_DIR / "masked_modeling_config.json"
    config = {
        'model_config': ClusterConfig.MASKED_CONFIG,
        'training_config': ClusterConfig.TRAINING_CONFIG,
        'training_time': str(training_time),
        'epochs_completed': trainer.current_epoch,
        'best_val_loss': float(trainer.callback_metrics.get('val_loss', float('inf')))
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Log final metrics to wandb
    wandb_logger.experiment.log({
        'final/training_time_seconds': training_time.total_seconds(),
        'final/epochs_completed': trainer.current_epoch,
        'final/best_val_loss': float(trainer.callback_metrics.get('val_loss', float('inf')))
    })
    
    return lightning_module, trainer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train SSL models on SLURM cluster')
    parser.add_argument('--model', choices=['temporal', 'masked', 'both'], default='both',
                        help='Which model(s) to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--temporal-epochs', type=int, default=None,
                        help='Epochs for temporal model (if different)')
    parser.add_argument('--masked-epochs', type=int, default=None,
                        help='Epochs for masked model (if different)')
    parser.add_argument('--wandb-project', type=str, default="hydro",
                        help='Wandb project name (overrides default)')
    parser.add_argument('--wandb-entity', type=str, default="tu-leopinetzki",
                        help='Wandb entity/username')
    parser.add_argument('--wandb-tags', type=str, nargs='*', default=[],
                        help='Additional wandb tags')
    
    args = parser.parse_args()
    
    # Setup
    setup_model_save_dir()
    
    # Update wandb config based on args
    if args.wandb_project:
        ClusterConfig.WANDB_CONFIG['project'] = args.wandb_project
    if args.wandb_entity:
        ClusterConfig.WANDB_CONFIG['entity'] = args.wandb_entity
    if args.wandb_tags:
        ClusterConfig.WANDB_CONFIG['tags'].extend(args.wandb_tags)
    
    # Set epoch counts
    temporal_epochs = args.temporal_epochs or args.epochs
    masked_epochs = args.masked_epochs or args.epochs
    
    logger.info("=" * 60)
    logger.info("SLURM CLUSTER SSL TRAINING STARTED")
    logger.info("=" * 60)
    logger.info(f"Model(s) to train: {args.model}")
    logger.info(f"Temporal epochs: {temporal_epochs}")
    logger.info(f"Masked epochs: {masked_epochs}")
    logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Model save directory: {ClusterConfig.MODEL_SAVE_DIR}")
    logger.info(f"Wandb project: {ClusterConfig.WANDB_CONFIG['project']}")
    logger.info(f"Wandb entity: {ClusterConfig.WANDB_CONFIG['entity'] or 'default'}")
    logger.info(f"Wandb tags: {ClusterConfig.WANDB_CONFIG['tags']}")
    
    # Load data
    try:
        precip_data = load_eobs_data()
    except Exception as e:
        logger.error(f"Failed to load E-OBS data: {e}")
        sys.exit(1)
    
    # Train models
    results = {}
    
    if args.model in ['temporal', 'both']:
        try:
            temporal_module, temporal_trainer = train_temporal_model(precip_data, temporal_epochs)
            results['temporal'] = {
                'completed': True,
                'epochs': temporal_trainer.current_epoch,
                'best_val_loss': float(temporal_trainer.callback_metrics.get('val_loss', float('inf')))
            }
        except Exception as e:
            logger.error(f"Temporal model training failed: {e}")
            results['temporal'] = {'completed': False, 'error': str(e)}
    
    if args.model in ['masked', 'both']:
        try:
            masked_module, masked_trainer = train_masked_model(precip_data, masked_epochs)
            results['masked'] = {
                'completed': True,
                'epochs': masked_trainer.current_epoch,
                'best_val_loss': float(masked_trainer.callback_metrics.get('val_loss', float('inf')))
            }
        except Exception as e:
            logger.error(f"Masked model training failed: {e}")
            results['masked'] = {'completed': False, 'error': str(e)}
    
    # Save final results
    results_path = ClusterConfig.MODEL_SAVE_DIR / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {results_path}")
    
    for model_name, result in results.items():
        if result['completed']:
            logger.info(f"{model_name.upper()}: ✅ {result['epochs']} epochs, val_loss: {result['best_val_loss']:.4f}")
        else:
            logger.info(f"{model_name.upper()}: ❌ Failed - {result['error']}")
    
    # Finish all wandb runs
    wandb.finish()


if __name__ == "__main__":
    main() 