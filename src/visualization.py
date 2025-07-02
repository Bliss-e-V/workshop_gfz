"""
Visualization utilities for the AI Workshop
CNN feature maps and Transformer attention visualization
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Tuple, Optional, List
import cv2


class CNNVisualizer:
    """Visualize CNN feature maps and activations"""
    
    def __init__(self, model):
        self.model = model
        
    def plot_feature_maps(self, feature_maps: Dict[str, torch.Tensor], 
                         num_channels: int = 8, figsize: Tuple[int, int] = (15, 10)):
        """Plot feature maps from different CNN layers"""
        num_layers = len(feature_maps)
        
        fig, axes = plt.subplots(num_layers, num_channels, figsize=figsize)
        fig.suptitle('🔍 CNN Feature Maps Visualization', fontsize=16, fontweight='bold')
        
        for layer_idx, (layer_name, features) in enumerate(feature_maps.items()):
            # Take first batch item
            feature_batch = features[0]  # Shape: [channels, H, W]
            
            for channel_idx in range(min(num_channels, feature_batch.shape[0])):
                ax = axes[layer_idx, channel_idx] if num_layers > 1 else axes[channel_idx]
                
                # Extract and normalize the feature map
                feature_map = feature_batch[channel_idx].cpu().numpy()
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                
                # Plot the feature map
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'{layer_name}\nCh.{channel_idx}', fontsize=8)
                ax.axis('off')
                
                # Add colorbar for the first channel of each layer
                if channel_idx == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    def plot_activation_statistics(self, feature_maps: Dict[str, torch.Tensor]):
        """Plot statistics of activations across layers"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('📊 CNN Activation Statistics', fontsize=16, fontweight='bold')
        
        layer_names = list(feature_maps.keys())
        
        # Mean activation per layer
        mean_activations = []
        std_activations = []
        sparsity_levels = []
        
        for layer_name, features in feature_maps.items():
            batch_features = features[0].cpu().numpy()  # First batch item
            mean_activations.append(np.mean(batch_features))
            std_activations.append(np.std(batch_features))
            sparsity_levels.append(np.mean(batch_features == 0) * 100)  # Percentage of zeros
        
        # Plot mean activations
        axes[0, 0].bar(layer_names, mean_activations, color='skyblue')
        axes[0, 0].set_title('Mean Activation per Layer')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot standard deviation
        axes[0, 1].bar(layer_names, std_activations, color='lightcoral')
        axes[0, 1].set_title('Activation Std per Layer')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot sparsity
        axes[1, 0].bar(layer_names, sparsity_levels, color='lightgreen')
        axes[1, 0].set_title('Sparsity per Layer (%)')
        axes[1, 0].set_ylabel('Percentage of Zeros')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Distribution of first layer activations
        first_layer_features = list(feature_maps.values())[0][0].cpu().numpy().flatten()
        axes[1, 1].hist(first_layer_features, bins=50, alpha=0.7, color='orange')
        axes[1, 1].set_title(f'{layer_names[0]} Activation Distribution')
        axes[1, 1].set_xlabel('Activation Value')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()


class TransformerVisualizer:
    """Visualize Transformer attention patterns"""
    
    def __init__(self, model):
        self.model = model
        
    def plot_attention_maps(self, attention_weights: Tuple, 
                           layer_idx: int = -1, head_idx: int = 0,
                           figsize: Tuple[int, int] = (12, 8)):
        """Plot attention maps from transformer"""
        if attention_weights is None:
            print("⚠️ No attention weights available")
            return
            
        # Get attention from specified layer
        attention = attention_weights[layer_idx][0]  # First batch item
        
        # attention shape: [num_heads, seq_len, seq_len]
        num_heads = attention.shape[0]
        seq_len = attention.shape[1]
        
        # Calculate grid size for subplot
        grid_size = min(4, num_heads)
        fig, axes = plt.subplots(2, grid_size, figsize=figsize)
        fig.suptitle(f'🎯 Attention Maps - Layer {layer_idx}', fontsize=16, fontweight='bold')
        
        for head in range(min(8, num_heads)):
            row = head // grid_size
            col = head % grid_size
            ax = axes[row, col] if num_heads > grid_size else axes[col]
            
            # Get attention matrix for this head
            att_matrix = attention[head].cpu().numpy()
            
            # Plot attention heatmap
            sns.heatmap(att_matrix, cmap='Blues', ax=ax, cbar=True, square=True)
            ax.set_title(f'Head {head}', fontsize=10)
            ax.set_xlabel('Key Positions')
            ax.set_ylabel('Query Positions')
        
        plt.tight_layout()
        plt.show()
    
    def plot_attention_rollout(self, attention_weights: Tuple, 
                              patch_size: int = 16, image_size: int = 224):
        """Create attention rollout visualization"""
        if attention_weights is None:
            print("⚠️ No attention weights available")
            return
            
        # Compute attention rollout
        rollout = self._compute_rollout(attention_weights)
        
        # Reshape to spatial dimensions
        num_patches = (image_size // patch_size) ** 2
        spatial_dim = int(np.sqrt(num_patches))
        
        # Remove CLS token and reshape
        spatial_attention = rollout[0, 1:].reshape(spatial_dim, spatial_dim)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw attention map
        im1 = ax1.imshow(spatial_attention.cpu().numpy(), cmap='viridis')
        ax1.set_title('🎯 Attention Rollout', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Upsampled version
        upsampled = cv2.resize(spatial_attention.cpu().numpy(), (image_size, image_size))
        im2 = ax2.imshow(upsampled, cmap='viridis')
        ax2.set_title('🔍 Upsampled Attention', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
        return upsampled
    
    def _compute_rollout(self, attention_weights: Tuple) -> torch.Tensor:
        """Compute attention rollout following Abnar & Zuidema (2020)"""
        result = torch.eye(attention_weights[0].shape[-1])
        
        for attention in attention_weights:
            # Average over heads
            attention_heads_fused = attention[0].mean(dim=0)  # First batch item
            
            # Add residual connection
            I = torch.eye(attention_heads_fused.size(-1))
            attention_heads_fused = attention_heads_fused + I
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Multiply with previous result
            result = torch.matmul(attention_heads_fused, result)
        
        return result


class ModelComparator:
    """Compare and visualize model performances"""
    
    def __init__(self):
        pass
    
    def plot_training_curves(self, cnn_history: Dict, vit_history: Dict):
        """Plot training curves for both models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📈 Training Comparison: CNN vs ViT', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(cnn_history['train_loss']) + 1)
        
        # Training Loss
        axes[0, 0].plot(epochs, cnn_history['train_loss'], 'b-', label='CNN', linewidth=2)
        axes[0, 0].plot(epochs, vit_history['train_loss'], 'r-', label='ViT', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation Loss
        axes[0, 1].plot(epochs, cnn_history['val_loss'], 'b-', label='CNN', linewidth=2)
        axes[0, 1].plot(epochs, vit_history['val_loss'], 'r-', label='ViT', linewidth=2)
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training Accuracy
        axes[1, 0].plot(epochs, cnn_history['train_acc'], 'b-', label='CNN', linewidth=2)
        axes[1, 0].plot(epochs, vit_history['train_acc'], 'r-', label='ViT', linewidth=2)
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation Accuracy
        axes[1, 1].plot(epochs, cnn_history['val_acc'], 'b-', label='CNN', linewidth=2)
        axes[1, 1].plot(epochs, vit_history['val_acc'], 'r-', label='ViT', linewidth=2)
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, cnn_predictions: np.ndarray, vit_predictions: np.ndarray,
                               true_labels: np.ndarray, class_names: List[str]):
        """Plot confusion matrices for both models"""
        from sklearn.metrics import confusion_matrix
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('🎯 Confusion Matrices: CNN vs ViT', fontsize=16, fontweight='bold')
        
        # CNN Confusion Matrix
        cnn_cm = confusion_matrix(true_labels, cnn_predictions)
        sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title('CNN Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # ViT Confusion Matrix
        vit_cm = confusion_matrix(true_labels, vit_predictions)
        sns.heatmap(vit_cm, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                   xticklabels=class_names, yticklabels=class_names)
        axes[1].set_title('ViT Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        plt.tight_layout()
        plt.show()


def overlay_attention_on_image(image: np.ndarray, attention_map: np.ndarray, 
                              alpha: float = 0.6) -> np.ndarray:
    """Overlay attention map on original image"""
    # Normalize attention map
    attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    # Create heatmap
    heatmap = plt.cm.jet(attention_normalized)[:, :, :3]  # Remove alpha channel
    
    # Overlay
    overlaid = alpha * heatmap + (1 - alpha) * image
    return np.clip(overlaid, 0, 1)


# --- Self-Supervised Learning Visualization Utilities -----------------------

def plot_temporal_prediction_results(model, dataloader, device=None, num_samples=4):
    """
    Visualize temporal prediction results
    
    Args:
        model: Trained temporal prediction model
        dataloader: DataLoader with temporal sequences
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    import torch
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Get a batch
        batch = next(iter(dataloader))
        inputs, targets = batch
        inputs = inputs.to(device)
        
        # Make predictions
        predictions = model(inputs).cpu()
        
        # Plot results
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('🔮 Temporal Prediction Results', fontsize=16, fontweight='bold')
        
        for i in range(num_samples):
            # Last input day
            ax = axes[i, 0]
            last_input = inputs[i, -1].cpu()
            im = ax.imshow(last_input, cmap='Blues')
            ax.set_title(f'Last Input Day {i+1}')
            ax.axis('off')
            
            # Prediction
            ax = axes[i, 1]
            pred = predictions[i, 0]
            im = ax.imshow(pred, cmap='Blues')
            ax.set_title(f'Prediction {i+1}')
            ax.axis('off')
            
            # Ground truth
            ax = axes[i, 2]
            target = targets[i, 0]
            im = ax.imshow(target, cmap='Blues')
            ax.set_title(f'Ground Truth {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and print metrics
        mse = torch.nn.functional.mse_loss(predictions, targets)
        mae = torch.nn.functional.l1_loss(predictions, targets)
        
        print(f"📊 Metrics:")
        print(f"   - MSE: {mse:.6f}")
        print(f"   - MAE: {mae:.6f}")


def plot_masked_modeling_results(model, dataloader, device=None, num_samples=4):
    """
    Visualize masked modeling reconstruction results
    
    Args:
        model: Trained masked modeling model
        dataloader: DataLoader with masked samples
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    import torch
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Get a batch
        batch = next(iter(dataloader))
        masked_inputs, targets, masks = batch
        masked_inputs = masked_inputs.to(device)
        
        # Make reconstructions
        reconstructions = model(masked_inputs).cpu()
        
        # Plot results
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('🎭 Masked Modeling Results', fontsize=16, fontweight='bold')
        
        for i in range(num_samples):
            # Original
            ax = axes[i, 0]
            original = targets[i, 0]
            im = ax.imshow(original, cmap='Blues')
            ax.set_title(f'Original {i+1}')
            ax.axis('off')
            
            # Masked input
            ax = axes[i, 1]
            masked = masked_inputs[i, 0].cpu()
            im = ax.imshow(masked, cmap='Blues')
            ax.set_title(f'Masked Input {i+1}')
            ax.axis('off')
            
            # Reconstruction
            ax = axes[i, 2]
            recon = reconstructions[i, 0]
            im = ax.imshow(recon, cmap='Blues')
            ax.set_title(f'Reconstruction {i+1}')
            ax.axis('off')
            
            # Mask overlay
            ax = axes[i, 3]
            mask = masks[i]
            ax.imshow(original, cmap='Blues', alpha=0.7)
            ax.imshow(mask, cmap='Reds', alpha=0.3)
            ax.set_title(f'Mask Overlay {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate metrics (only on masked regions)
        mask_expanded = masks.unsqueeze(1).float()
        masked_mse = torch.nn.functional.mse_loss(
            reconstructions * mask_expanded, 
            targets * mask_expanded
        )
        masked_mae = torch.nn.functional.l1_loss(
            reconstructions * mask_expanded, 
            targets * mask_expanded
        )
        
        print(f"📊 Metrics (masked regions only):")
        print(f"   - MSE: {masked_mse:.6f}")
        print(f"   - MAE: {masked_mae:.6f}")


def plot_training_comparison(temporal_trainer, masked_trainer):
    """
    Compare training curves of both self-supervised approaches
    
    Args:
        temporal_trainer: Trained temporal prediction trainer
        masked_trainer: Trained masked modeling trainer
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Temporal prediction curves
    ax1.plot(temporal_trainer.train_losses, label='Training Loss', color='blue')
    ax1.plot(temporal_trainer.val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('🔮 Temporal Prediction Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Masked modeling curves
    ax2.plot(masked_trainer.train_losses, label='Training Loss', color='blue')
    ax2.plot(masked_trainer.val_losses, label='Validation Loss', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('🎭 Masked Modeling Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final results
    print(f"🔮 Temporal Prediction - Final Val Loss: {temporal_trainer.val_losses[-1]:.6f}")
    print(f"🎭 Masked Modeling - Final Val Loss: {masked_trainer.val_losses[-1]:.6f}")


def plot_self_supervised_concept():
    """
    Create a conceptual diagram explaining self-supervised learning approaches
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Self-Supervised Learning Approaches for Climate Data', fontsize=16, fontweight='bold')
    
    # Temporal Prediction Concept
    ax1.set_title('🔮 Temporal Prediction', fontsize=14, fontweight='bold')
    
    # Create dummy temporal sequence
    days = np.arange(8)
    pattern = np.sin(days * 0.5) + np.random.normal(0, 0.1, 8)
    
    ax1.plot(days[:-1], pattern[:-1], 'bo-', label='Input Sequence (7 days)', linewidth=2, markersize=8)
    ax1.plot(days[-1], pattern[-1], 'ro', label='Target (Day 8)', markersize=12)
    ax1.axvline(x=6.5, color='gray', linestyle='--', alpha=0.7)
    ax1.text(3, max(pattern), 'Past 7 days', ha='center', fontsize=12, fontweight='bold')
    ax1.text(7, max(pattern), 'Predict', ha='center', fontsize=12, fontweight='bold', color='red')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Precipitation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temporal Architecture
    ax2.set_title('ConvLSTM Architecture', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.8, 'Input: (7, H, W)', ha='center', va='center', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), fontsize=12)
    ax2.text(0.5, 0.6, 'ConvLSTM\nLayers', ha='center', va='center', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"), fontsize=12)
    ax2.text(0.5, 0.4, 'Output Conv', ha='center', va='center', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsalmon"), fontsize=12)
    ax2.text(0.5, 0.2, 'Output: (1, H, W)', ha='center', va='center', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"), fontsize=12)
    
    # Add arrows
    ax2.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.75), 
                arrowprops=dict(arrowstyle='->', lw=2), transform=ax2.transAxes)
    ax2.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.55), 
                arrowprops=dict(arrowstyle='->', lw=2), transform=ax2.transAxes)
    ax2.annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.35), 
                arrowprops=dict(arrowstyle='->', lw=2), transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Masked Modeling Concept
    ax3.set_title('🎭 Masked Modeling', fontsize=14, fontweight='bold')
    
    # Create dummy spatial data
    x, y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    z = np.sin(5*x) * np.cos(5*y) + 0.1*np.random.randn(20, 20)
    
    # Create mask
    mask = np.zeros_like(z, dtype=bool)
    mask[5:15, 8:12] = True  # Block mask
    mask[2:8, 2:6] = True    # Another block
    
    z_masked = z.copy()
    z_masked[mask] = 0
    
    im = ax3.imshow(z_masked, cmap='Blues', extent=[0, 1, 0, 1])
    ax3.imshow(mask, cmap='Reds', alpha=0.5, extent=[0, 1, 0, 1])
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.text(0.5, -0.1, 'Masked regions (red) to be reconstructed', ha='center', 
             transform=ax3.transAxes, fontsize=12, fontweight='bold')
    
    # U-Net Architecture
    ax4.set_title('U-Net Architecture', fontsize=14, fontweight='bold')
    
    # Draw U-Net structure
    levels = ['Input', 'Encoder 1', 'Encoder 2', 'Bottleneck', 'Decoder 2', 'Decoder 1', 'Output']
    y_pos = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    x_pos = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    
    # Draw boxes
    for i, (level, y, x) in enumerate(zip(levels, y_pos, x_pos)):
        if 'Encoder' in level or level == 'Input':
            color = 'lightblue'
        elif 'Decoder' in level or level == 'Output':
            color = 'lightcoral'
        else:
            color = 'lightgreen'
        
        ax4.text(x, y, level, ha='center', va='center', transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color), fontsize=10)
    
    # Draw arrows (simplified)
    for i in range(len(levels)-1):
        if i < 3:  # Encoder path
            ax4.annotate('', xy=(x_pos[i+1]-0.05, y_pos[i+1]), xytext=(x_pos[i]+0.05, y_pos[i]), 
                        arrowprops=dict(arrowstyle='->', lw=1.5), transform=ax4.transAxes)
        else:  # Decoder path
            ax4.annotate('', xy=(x_pos[i+1]-0.05, y_pos[i+1]), xytext=(x_pos[i]+0.05, y_pos[i]), 
                        arrowprops=dict(arrowstyle='->', lw=1.5), transform=ax4.transAxes)
    
    # Skip connections
    ax4.annotate('', xy=(0.65, 0.6), xytext=(0.25, 0.25), 
                arrowprops=dict(arrowstyle='->', lw=1, linestyle='--', color='gray'), 
                transform=ax4.transAxes)
    ax4.annotate('', xy=(0.75, 0.75), xytext=(0.15, 0.1), 
                arrowprops=dict(arrowstyle='->', lw=1, linestyle='--', color='gray'), 
                transform=ax4.transAxes)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("🎯 Self-Supervised Learning Benefits:")
    print("   • No labeled data required")
    print("   • Learns meaningful representations")
    print("   • Scalable to large datasets")
    print("   • Transferable to downstream tasks") 