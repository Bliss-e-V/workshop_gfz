"""
Model implementations for the AI Workshop
CNN vs Vision Transformer comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Tuple, Dict, Optional
import pytorch_lightning as L


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
    Using torchvision's ViT implementation
    """
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super(SatelliteViT, self).__init__()
        
        self.num_classes = num_classes
        
        if pretrained:
            # Use pre-trained ViT-Base from torchvision
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            # Replace the classification head
            self.vit.heads = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Linear(768, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        else:
            # Create from scratch for demonstration
            self.vit = vit_b_16(weights=None)
            # Replace the classification head
            self.vit.heads = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Linear(768, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        
        # Store attention weights for visualization (simplified for torchvision)
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through ViT
        # Note: torchvision's ViT doesn't expose attention weights as easily
        # For attention visualization, we'd need to register hooks or use a different approach
        logits = self.vit(x)
        return logits
    
    def get_attention_weights(self) -> Optional[Tuple]:
        """Return attention weights for visualization"""
        # Note: torchvision's ViT doesn't expose attention weights by default
        # This would require registering hooks to capture attention from the transformer layers
        return None


class FloodViT(nn.Module):
    """
    Lightweight Vision Transformer for binary flood detection
    Uses torchvision's ViT-Base but with binary classification head
    """
    
    def __init__(self, pretrained: bool = True):
        super(FloodViT, self).__init__()
        
        self.num_classes = 2
        
        if pretrained:
            # Start with pre-trained ViT-Base from torchvision
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            # Replace with binary classification head
            self.vit.heads = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Linear(768, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2)  # Binary output
            )
        else:
            # Create from scratch for demonstration
            self.vit = vit_b_16(weights=None)
            # Replace with binary classification head
            self.vit.heads = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Linear(768, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2)  # Binary output
            )
        
        # Store attention weights for visualization (simplified for torchvision)
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through ViT
        # Note: torchvision's ViT doesn't expose attention weights as easily
        # For attention visualization, we'd need to register hooks or use a different approach
        logits = self.vit(x)
        return logits
    
    def get_attention_weights(self) -> Optional[Tuple]:
        """Return attention weights for visualization"""
        # Note: torchvision's ViT doesn't expose attention weights by default
        # This would require registering hooks to capture attention from the transformer layers
        return None


class ModelComparator:
    """Utility class for comparing different models"""
    
    def __init__(self, cnn_model, vit_model):
        self.cnn_model = cnn_model
        self.vit_model = vit_model
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in both models"""
        cnn_params = sum(p.numel() for p in self.cnn_model.parameters())
        vit_params = sum(p.numel() for p in self.vit_model.parameters())
        
        return {
            'cnn_parameters': cnn_params,
            'vit_parameters': vit_params,
            'ratio': vit_params / cnn_params if cnn_params > 0 else 0
        }
    
    def get_model_info(self) -> Dict[str, Dict]:
        """Get detailed information about both models"""
        return {
            'cnn': {
                'name': self.cnn_model.__class__.__name__,
                'parameters': sum(p.numel() for p in self.cnn_model.parameters()),
                'trainable': sum(p.numel() for p in self.cnn_model.parameters() if p.requires_grad),
            },
            'vit': {
                'name': self.vit_model.__class__.__name__,
                'parameters': sum(p.numel() for p in self.vit_model.parameters()),
                'trainable': sum(p.numel() for p in self.vit_model.parameters() if p.requires_grad),
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
    """Generate a summary of model architecture"""
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_params(model)
    
    summary = f"""
    Model: {model.__class__.__name__}
    Total trainable parameters: {total_params:,}
    Input size: {input_size}
    """
    
    return summary 


# --- Self-Supervised Learning Models for E-OBS Climate Data ---------------------

class TemporalPredictionModel(nn.Module):
    """
    Model for temporal prediction of precipitation using ConvLSTM architecture.
    Predicts future precipitation patterns from past sequences.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,  # Reduced from 64 to 32 for faster training
        num_layers: int = 2,  # Reduced from 3 to 2 for faster training
        sequence_length: int = 7,
        prediction_horizon: int = 1,
        spatial_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.spatial_size = spatial_size
        
        # ConvLSTM layers for temporal modeling
        self.convlstm_layers = nn.ModuleList()
        
        # First layer
        self.convlstm_layers.append(
            ConvLSTMCell(input_channels, hidden_channels, kernel_size=3)
        )
        
        # Additional layers
        for i in range(1, num_layers):
            self.convlstm_layers.append(
                ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=3)
            )
        
        # Output layer for prediction
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, prediction_horizon, kernel_size=1),
        )
        
        # Determine if this is optimized for speed or full capacity
        is_speed_optimized = hidden_channels <= 32 and num_layers <= 2
        config_type = "OPTIMIZED FOR SPEED" if is_speed_optimized else "FULL CAPACITY"
        
        print(f"🔮 TemporalPredictionModel initialized ({config_type}):")
        print(f"   - Input: ({sequence_length}, {input_channels}, {spatial_size[0]}, {spatial_size[1]})")
        print(f"   - Output: ({prediction_horizon}, {spatial_size[0]}, {spatial_size[1]})")
        print(f"   - Hidden channels: {hidden_channels}")
        print(f"   - Layers: {num_layers}")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, height, width)
        
        Returns:
            predictions: Tensor of shape (batch_size, prediction_horizon, height, width)
        """
        batch_size, seq_len, height, width = x.shape
        
        # Initialize hidden states for all layers
        hidden_states = []
        cell_states = []
        
        for layer in self.convlstm_layers:
            h, c = layer.init_hidden(batch_size, height, width, x.device)
            hidden_states.append(h)
            cell_states.append(c)
        
        # Process sequence
        for t in range(seq_len):
            input_t = x[:, t:t + 1, :, :]  # (batch_size, 1, H, W)
            
            # Pass through ConvLSTM layers
            for i, layer in enumerate(self.convlstm_layers):
                if i == 0:
                    hidden_states[i], cell_states[i] = layer(
                        input_t, (hidden_states[i], cell_states[i])
                    )
                else:
                    hidden_states[i], cell_states[i] = layer(
                        hidden_states[i - 1], (hidden_states[i], cell_states[i])
                    )
        
        # Generate predictions from final hidden state
        final_hidden = hidden_states[-1]  # (batch_size, hidden_channels, H, W)
        predictions = self.output_conv(final_hidden)  # (batch_size, pred_horizon, H, W)
        
        return predictions


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for spatio-temporal modeling"""
    
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Combined convolution for all gates
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,  # i, f, o, g gates
            kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Apply convolution
        combined_conv = self.conv(combined)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        
        # Apply gate functions
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # New gate
        
        # Update cell state
        c_next = f * c_cur + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width, device):
        """Initialize hidden and cell states"""
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class MaskedModelingModel(nn.Module):
    """
    U-Net style model for masked precipitation reconstruction.
    Takes masked precipitation maps and reconstructs the original.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,  # Reduced from 64 to 32 for faster training
        num_levels: int = 3,  # Reduced from 4 to 3 for faster training
        temporal_context: int = 1
    ):
        super().__init__()
        
        self.input_channels = input_channels * temporal_context
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.temporal_context = temporal_context
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        in_channels = self.input_channels
        for i in range(num_levels):
            out_channels = base_channels * (2 ** i)
            self.encoder_blocks.append(
                self._make_conv_block(in_channels, out_channels)
            )
            in_channels = out_channels
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(
            in_channels, in_channels * 2
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()
        
        in_channels = in_channels * 2
        for i in range(num_levels):
            out_channels = base_channels * (2 ** (num_levels - i - 1))
            
            # Upsampling
            self.upconv_blocks.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            
            # Decoder block (with skip connection)
            self.decoder_blocks.append(
                self._make_conv_block(out_channels * 2, out_channels)
            )
            
            in_channels = out_channels
        
        # Final output layer with proper initialization
        self.output_conv = nn.Conv2d(
            base_channels, temporal_context, kernel_size=1, bias=True
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Determine if this is optimized for speed or full capacity
        is_speed_optimized = base_channels <= 32 and num_levels <= 3
        config_type = "OPTIMIZED FOR SPEED" if is_speed_optimized else "FULL CAPACITY"
        
        print(f"🎭 MaskedModelingModel initialized ({config_type}):")
        print(f"   - Input channels: {self.input_channels}")
        print(f"   - Base channels: {base_channels}")
        print(f"   - Levels: {num_levels}")
        print(f"   - Temporal context: {temporal_context}")
    
    def _initialize_weights(self):
        """Initialize weights for numerical stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with two conv layers (numerically stable)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=False),  # Use inplace=False for stability
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=False)   # Use inplace=False for stability
        )
    
    def forward(self, x):
        """
        Args:
            x: Masked input tensor (batch_size, temporal_context, height, width)
        
        Returns:
            reconstruction: Reconstructed tensor (batch_size, temporal_context, height, width)
        """
        # Encoder path
        encoder_features = []
        current = x
        
        for i, encoder_block in enumerate(self.encoder_blocks):
            current = encoder_block(current)
            encoder_features.append(current)
            if i < len(self.encoder_blocks) - 1:  # Don't pool after last encoder
                current = self.pool(current)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_blocks, self.decoder_blocks)):
            # Upsample
            current = upconv(current)
            
            # Skip connection
            skip_features = encoder_features[-(i + 1)]
            
            # Handle size mismatch due to pooling (with stability check)
            if current.shape != skip_features.shape:
                current = F.interpolate(
                    current, size=skip_features.shape[2:], mode='bilinear', align_corners=False
                )
                
                # Check for NaN after interpolation
                if torch.isnan(current).any():
                    print(f"⚠️ NaN detected after interpolation in decoder layer {i}")
                    current = torch.nan_to_num(current, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Concatenate skip connection
            current = torch.cat([current, skip_features], dim=1)
            
            # Apply decoder block
            current = decoder_block(current)
        
        # Final output
        reconstruction = self.output_conv(current)
        
        return reconstruction


class TemporalPredictionLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for temporal prediction self-supervised learning.
    Uses past 7 days to predict the next day's precipitation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        print("🚀 TemporalPredictionLightningModule initialized:")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Weight decay: {weight_decay}")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.model(inputs)
        loss = self.criterion(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.model(inputs)
        loss = self.criterion(predictions, targets)
        
        # Calculate additional metrics
        mse = F.mse_loss(predictions, targets)
        mae = F.l1_loss(predictions, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse, on_step=False, on_epoch=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


class MaskedModelingLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for masked modeling self-supervised learning.
    Reconstructs masked spatial regions of precipitation maps.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function (no reduction for mask-specific loss)
        self.criterion = nn.MSELoss(reduction='none')
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        print("🚀 MaskedModelingLightningModule initialized:")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Weight decay: {weight_decay}")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        masked_inputs, targets, masks = batch
        reconstructions = self.model(masked_inputs)
        
        # Calculate loss only on masked regions
        loss_map = self.criterion(reconstructions, targets)
        mask_expanded = masks.unsqueeze(1).float()
        masked_loss = loss_map * mask_expanded
        
        # Prevent division by zero by adding small epsilon
        num_masked_pixels = mask_expanded.sum() + 1e-8
        loss = masked_loss.sum() / num_masked_pixels
        
        # Check for NaN and clip loss if needed
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf detected in training step {batch_idx}")
            print(f"   Mask sum: {masks.sum()}, Loss map stats: min={loss_map.min():.6f}, max={loss_map.max():.6f}")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        masked_inputs, targets, masks = batch
        reconstructions = self.model(masked_inputs)
        
        # Calculate loss only on masked regions
        loss_map = self.criterion(reconstructions, targets)
        mask_expanded = masks.unsqueeze(1).float()
        masked_loss = loss_map * mask_expanded
        
        # Prevent division by zero by adding small epsilon
        num_masked_pixels = mask_expanded.sum() + 1e-8
        loss = masked_loss.sum() / num_masked_pixels
        
        # Check for NaN and handle gracefully
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf detected in validation step {batch_idx}")
            print(f"   Mask sum: {masks.sum()}, Loss map stats: min={loss_map.min():.6f}, max={loss_map.max():.6f}")
            loss = torch.tensor(0.0, device=loss.device)
        
        # Calculate additional metrics on masked regions only (more robust)
        masked_reconstructions = reconstructions * mask_expanded
        masked_targets = targets * mask_expanded
        
        # Only calculate metrics if there are masked pixels
        if mask_expanded.sum() > 0:
            mse_numerator = F.mse_loss(masked_reconstructions, masked_targets, reduction='sum')
            mae_numerator = F.l1_loss(masked_reconstructions, masked_targets, reduction='sum')
            denominator = mask_expanded.sum() + 1e-8
            masked_mse = mse_numerator / denominator
            masked_mae = mae_numerator / denominator
        else:
            masked_mse = torch.tensor(0.0, device=loss.device)
            masked_mae = torch.tensor(0.0, device=loss.device)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_masked_mse', masked_mse, on_step=False, on_epoch=True)
        self.log('val_masked_mae', masked_mae, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


class SelfSupervisedTrainer:
    """
    Legacy trainer class - kept for backwards compatibility.
    Use Lightning modules instead for new code.
    """
    
    def __init__(self, *args, **kwargs):
        print("⚠️  Warning: SelfSupervisedTrainer is deprecated.")
        print("   Please use TemporalPredictionLightningModule or MaskedModelingLightningModule instead.")
        print("   This class is kept for backwards compatibility only.")
        
        # Initialize with minimal functionality
        self.train_losses = []
        self.val_losses = [] 