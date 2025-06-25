"""
Model implementations for the AI Workshop
CNN vs Vision Transformer comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from typing import Tuple, Dict, Optional
import numpy as np


class SatelliteCNN(nn.Module):
    """
    Custom CNN for satellite image classification
    Designed for EuroSAT dataset with 10 classes
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(SatelliteCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)
        )
        
        # Store intermediate activations for visualization
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        self.features[6].register_forward_hook(get_activation('block1'))
        self.features[13].register_forward_hook(get_activation('block2'))
        self.features[20].register_forward_hook(get_activation('block3'))
        self.features[27].register_forward_hook(get_activation('block4'))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        """Return stored feature maps for visualization"""
        return self.activations


class FloodCNN(nn.Module):
    """
    Lightweight CNN optimized for binary flood detection
    Simpler architecture appropriate for water/no-water classification
    """
    
    def __init__(self, dropout_rate: float = 0.3):
        super(FloodCNN, self).__init__()
        
        self.num_classes = 2  # Binary classification
        
        # Lighter feature extraction - fewer parameters
        self.features = nn.Sequential(
            # Block 1 - Focus on water edge detection
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2 - Texture patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3 - Higher-level features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4 - Compact representation
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce spatial dimensions
        )
        
        # Simpler classifier for binary decision
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)  # Binary output
        )
        
        # Store intermediate activations for visualization
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        self.features[2].register_forward_hook(get_activation('block1'))
        self.features[5].register_forward_hook(get_activation('block2'))
        self.features[8].register_forward_hook(get_activation('block3'))
        self.features[11].register_forward_hook(get_activation('block4'))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        """Return stored feature maps for visualization"""
        return self.activations


class SatelliteViT(nn.Module):
    """
    Vision Transformer for satellite image classification
    Using HuggingFace transformers with custom head
    """
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super(SatelliteViT, self).__init__()
        
        self.num_classes = num_classes
        
        if pretrained:
            # Use pre-trained ViT-Base
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            # Create from scratch for demonstration
            config = ViTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=224,
                patch_size=16,
                num_channels=3,
                num_labels=num_classes
            )
            self.vit = ViTModel(config)
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get ViT outputs with attention weights
        outputs = self.vit(x, output_attentions=True)
        
        # Store attention weights for visualization
        self.attention_weights = outputs.attentions
        
        # Use CLS token for classification
        cls_output = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(cls_output)
        
        return logits
    
    def get_attention_weights(self) -> Optional[Tuple]:
        """Return attention weights for visualization"""
        return self.attention_weights


class FloodViT(nn.Module):
    """
    Lightweight Vision Transformer for binary flood detection
    Uses fewer transformer layers and smaller hidden dimensions
    """
    
    def __init__(self, pretrained: bool = True):
        super(FloodViT, self).__init__()
        
        self.num_classes = 2
        
        if pretrained:
            # Start with pre-trained but adapt for binary classification
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            # Smaller config for binary classification
            config = ViTConfig(
                hidden_size=384,  # Smaller hidden size
                num_hidden_layers=6,  # Fewer layers
                num_attention_heads=6,
                intermediate_size=1536,
                image_size=224,
                patch_size=16,
                num_channels=3,
                num_labels=2
            )
            self.vit = ViTModel(config)
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.vit.config.hidden_size, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # Binary output
        )
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get ViT outputs with attention weights
        outputs = self.vit(x, output_attentions=True)
        
        # Store attention weights for visualization
        self.attention_weights = outputs.attentions
        
        # Use CLS token for classification
        cls_output = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(cls_output)
        
        return logits
    
    def get_attention_weights(self) -> Optional[Tuple]:
        """Return attention weights for visualization"""
        return self.attention_weights


class ModelComparator:
    """Utility class to compare CNN and ViT models"""
    
    def __init__(self, cnn_model, vit_model):
        self.cnn_model = cnn_model
        self.vit_model = vit_model
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters for both models"""
        cnn_params = sum(p.numel() for p in self.cnn_model.parameters() if p.requires_grad)
        vit_params = sum(p.numel() for p in self.vit_model.parameters() if p.requires_grad)
        
        return {
            'CNN': cnn_params,
            'ViT': vit_params,
            'Ratio (ViT/CNN)': vit_params / cnn_params
        }
    
    def get_model_info(self) -> Dict[str, Dict]:
        """Get detailed information about both models"""
        param_counts = self.count_parameters()
        
        return {
            'CNN': {
                'Architecture': f'Custom CNN ({self.cnn_model.__class__.__name__})',
                'Parameters': f"{param_counts['CNN']:,}",
                'Key Features': ['Convolutional layers', 'Spatial feature maps', 'Translation invariance']
            },
            'ViT': {
                'Architecture': f'Vision Transformer ({self.vit_model.__class__.__name__})',
                'Parameters': f"{param_counts['ViT']:,}",
                'Key Features': ['Self-attention', 'Patch embeddings', 'Global context']
            }
        }


def create_models(num_classes: int = 10, task_type: str = "landcover") -> Tuple:
    """Factory function to create appropriate models based on task type"""
    
    if task_type == "flood":
        print("🌊 Creating models optimized for binary flood detection...")
        
        # Lightweight CNN for flood detection
        cnn = FloodCNN(dropout_rate=0.3)
        print(f"   ✅ FloodCNN: {sum(p.numel() for p in cnn.parameters()):,} parameters")
        
        # Lightweight ViT for flood detection
        try:
            vit = FloodViT(pretrained=True)
            print("   ✅ FloodViT with pre-trained backbone")
        except Exception as e:
            print(f"   ⚠️ Using FloodViT without pre-trained weights: {e}")
            vit = FloodViT(pretrained=False)
        
        print(f"   ✅ FloodViT: {sum(p.numel() for p in vit.parameters()):,} parameters")
        
    else:  # landcover
        print("🌍 Creating models optimized for multi-class land-cover classification...")
        
        # Full CNN for land-cover
        cnn = SatelliteCNN(num_classes=num_classes)
        print(f"   ✅ SatelliteCNN: {sum(p.numel() for p in cnn.parameters()):,} parameters")
        
        # Full ViT for land-cover
        try:
            vit = SatelliteViT(num_classes=num_classes, pretrained=True)
            print("   ✅ SatelliteViT with pre-trained weights")
        except Exception as e:
            print(f"   ⚠️ Using SatelliteViT without pre-trained weights: {e}")
            vit = SatelliteViT(num_classes=num_classes, pretrained=False)
        
        print(f"   ✅ SatelliteViT: {sum(p.numel() for p in vit.parameters()):,} parameters")
    
    return cnn, vit


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int]) -> str:
    """Get a summary of model architecture"""
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_params(model)
    
    summary = f"""
    Model: {model.__class__.__name__}
    Total trainable parameters: {total_params:,}
    Input size: {input_size}
    """
    
    return summary 