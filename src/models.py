"""
Model implementations for Self-Supervised Learning with E-OBS data
"""

import torch
import torch.nn as nn
from typing import Tuple
import pytorch_lightning as L


class TemporalPredictionModel(nn.Module):
    """ConvLSTM-based model for temporal prediction of precipitation"""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,  # Used as embed_dim
        num_layers: int = 2,
        sequence_length: int = 7,
        prediction_horizon: int = 1,
        spatial_size: Tuple[int, int] = (64, 64),
    ):
        """
        Initialize temporal prediction model

        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels/features
            num_layers: Number of ConvLSTM layers
            sequence_length: Length of input sequence
            prediction_horizon: Number of time steps to predict
            spatial_size: Spatial dimensions (height, width)
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.spatial_size = spatial_size

        # Patch embedding parameters
        self.patch_size = 8
        self.num_patches_h = spatial_size[0] // self.patch_size
        self.num_patches_w = spatial_size[1] // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_channels))

        # Temporal transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                LightweightTransformerBlock(
                    embed_dim=hidden_channels, num_heads=4, mlp_ratio=4.0, dropout=0.1
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_channels)

        # Prediction head
        self.prediction_head = nn.Linear(
            hidden_channels, input_channels * (self.patch_size**2)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def patchify(self, x):
        """
        Convert images to patches
        Args:
            x: (batch_size, time, channels, height, width)
        Returns:
            patches: (batch_size, time, num_patches, patch_dim)
        """
        batch_size, time, channels, height, width = x.shape

        # Reshape to process all time steps together
        x = x.view(batch_size * time, channels, height, width)

        # Apply patch embedding
        patches = self.patch_embed(
            x
        )  # (batch_size * time, hidden_channels, num_patches_h, num_patches_w)

        # Flatten spatial dimensions
        patches = patches.flatten(2).transpose(
            1, 2
        )  # (batch_size * time, num_patches, hidden_channels)

        # Reshape back to include time dimension
        patches = patches.view(batch_size, time, self.num_patches, self.hidden_channels)

        return patches

    def unpatchify(self, patches):
        """
        Convert patches back to images
        Args:
            patches: (batch_size, time, num_patches, patch_dim)
        Returns:
            x: (batch_size, time, channels, height, width)
        """
        batch_size, time, num_patches, patch_dim = patches.shape

        # Reshape patches for reconstruction
        patches = patches.view(batch_size * time, num_patches, patch_dim)

        # Apply prediction head
        patches = self.prediction_head(
            patches
        )  # (batch_size * time, num_patches, channels * patch_size^2)

        # Reshape to spatial dimensions
        channels = self.input_channels
        patches = patches.view(
            batch_size * time,
            self.num_patches_h,
            self.num_patches_w,
            channels,
            self.patch_size,
            self.patch_size,
        )

        # Rearrange to image format
        x = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(
            batch_size * time,
            channels,
            self.num_patches_h * self.patch_size,
            self.num_patches_w * self.patch_size,
        )

        # Reshape back to include time dimension
        x = x.view(
            batch_size, time, channels, self.spatial_size[0], self.spatial_size[1]
        )

        return x

    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, sequence_length, channels, height, width)
        Returns:
            predictions: (batch_size, prediction_horizon, channels, height, width)
        """
        batch_size, sequence_length, channels, height, width = x.shape

        # Convert to patches
        patches = self.patchify(
            x
        )  # (batch_size, sequence_length, num_patches, hidden_channels)

        # Add positional encoding
        patches = patches + self.pos_embed.unsqueeze(
            1
        )  # Add time dimension to pos_embed

        # Process each sequence position
        predictions = []
        pred_patches = None

        for t in range(self.prediction_horizon):
            # Get current context (last sequence_length time steps)
            if t == 0:
                context = patches  # Use input sequence
            else:
                # Use a mix of input and previous predictions
                context = torch.cat([patches[:, 1:], pred_patches.unsqueeze(1)], dim=1)

            # Prepare input for transformer (flatten time and patch dimensions)
            context_flat = context.reshape(
                batch_size, sequence_length * self.num_patches, self.hidden_channels
            )

            # Apply transformer layers
            for layer in self.transformer_layers:
                context_flat = layer(context_flat)

            # Apply layer norm
            context_flat = self.norm(context_flat)

            # Get prediction for next time step (use last time step's patches)
            pred_patches = context_flat[
                :, -self.num_patches :, :
            ]  # (batch_size, num_patches, hidden_channels)

            # Convert back to spatial format
            pred_spatial = self.unpatchify(
                pred_patches.unsqueeze(1)
            )  # Add time dimension
            pred_spatial = pred_spatial.squeeze(1)  # Remove time dimension

            predictions.append(pred_spatial)

        # Stack predictions
        predictions = torch.stack(
            predictions, dim=1
        )  # (batch_size, prediction_horizon, channels, height, width)

        return predictions


class LightweightTransformerBlock(nn.Module):
    """Lightweight transformer block for temporal modeling"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        """
        Initialize transformer block

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, sequence_length, embed_dim)
        Returns:
            x: (batch_size, sequence_length, embed_dim)
        """
        # Multi-head attention with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class MaskedModelingModel(nn.Module):
    """U-Net based model for masked modeling of precipitation"""

    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,  # Reduced from 64 to 32 for faster training
        num_levels: int = 3,  # Reduced from 4 to 3 for faster training
        temporal_context: int = 1,
    ):
        """
        Initialize masked modeling model

        Args:
            input_channels: Number of input channels
            base_channels: Base number of channels
            num_levels: Number of encoder/decoder levels
            temporal_context: Number of time steps to process
        """
        super().__init__()

        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.temporal_context = temporal_context

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        in_channels = input_channels
        for i in range(num_levels):
            out_channels = base_channels * (2**i)
            self.encoder.append(self._make_conv_block(in_channels, out_channels))
            if i < num_levels - 1:  # No pooling for the last level
                self.encoder_pools.append(nn.MaxPool2d(2))
            in_channels = out_channels

        # Decoder
        self.decoder = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        for i in range(num_levels - 1):
            level = num_levels - 2 - i
            in_channels = base_channels * (2 ** (level + 1))
            out_channels = base_channels * (2**level)

            self.decoder_upsamples.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            self.decoder.append(
                self._make_conv_block(
                    out_channels * 2, out_channels
                )  # *2 for skip connection
            )

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, input_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            reconstruction: (batch_size, channels, height, width)
        """
        # Encoder
        encoder_features = []

        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            encoder_features.append(x)

            if i < len(self.encoder_pools):
                x = self.encoder_pools[i](x)

        # Decoder
        for i, (upsample, decoder_block) in enumerate(
            zip(self.decoder_upsamples, self.decoder)
        ):
            x = upsample(x)

            # Skip connection
            skip_feature = encoder_features[
                -(i + 2)
            ]  # Get corresponding encoder feature
            x = torch.cat([x, skip_feature], dim=1)

            x = decoder_block(x)

        # Final output
        x = self.final_conv(x)

        return x


class TemporalPredictionLightningModule(L.LightningModule):
    """PyTorch Lightning module for temporal prediction"""

    def __init__(
        self, model: nn.Module, learning_rate: float = 1e-3, weight_decay: float = 1e-4
    ):
        """
        Initialize Lightning module

        Args:
            model: The temporal prediction model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss function
        self.criterion = nn.MSELoss()

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        sequences, targets = batch
        predictions = self.model(sequences)

        loss = self.criterion(predictions, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        sequences, targets = batch
        predictions = self.model(sequences)

        loss = self.criterion(predictions, targets)

        # Additional metrics
        mae = torch.mean(torch.abs(predictions - targets))
        mse = torch.mean((predictions - targets) ** 2)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class MaskedModelingLightningModule(L.LightningModule):
    """PyTorch Lightning module for masked modeling"""

    def __init__(
        self, model: nn.Module, learning_rate: float = 1e-3, weight_decay: float = 1e-4
    ):
        """
        Initialize Lightning module

        Args:
            model: The masked modeling model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss function
        self.criterion = nn.MSELoss()

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        masked_inputs, targets, masks = batch
        predictions = self.model(masked_inputs)

        # Calculate loss only on masked regions
        # masks: 1 for visible regions, 0 for masked regions
        # We want to compute loss only on masked regions (where mask == 0)
        mask_loss_weight = (1 - masks).unsqueeze(1)  # Add channel dimension

        # Weighted loss focusing on masked regions
        loss = torch.mean(mask_loss_weight * (predictions - targets) ** 2)

        # Also compute full reconstruction loss for monitoring
        full_loss = self.criterion(predictions, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_full_loss", full_loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        masked_inputs, targets, masks = batch
        predictions = self.model(masked_inputs)

        # Calculate loss only on masked regions
        mask_loss_weight = (1 - masks).unsqueeze(1)  # Add channel dimension

        # Weighted loss focusing on masked regions
        loss = torch.mean(mask_loss_weight * (predictions - targets) ** 2)

        # Additional metrics
        # MAE on masked regions only
        masked_mae = torch.mean(mask_loss_weight * torch.abs(predictions - targets))

        # Full reconstruction metrics
        full_loss = self.criterion(predictions, targets)
        full_mae = torch.mean(torch.abs(predictions - targets))

        # Reconstruction quality metrics
        # PSNR-like metric (higher is better)
        mse_masked = torch.mean(mask_loss_weight * (predictions - targets) ** 2)
        psnr_masked = 20 * torch.log10(targets.max() / (torch.sqrt(mse_masked) + 1e-8))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_masked_mae", masked_mae, on_step=False, on_epoch=True)
        self.log("val_full_loss", full_loss, on_step=False, on_epoch=True)
        self.log("val_full_mae", full_mae, on_step=False, on_epoch=True)
        self.log("val_psnr_masked", psnr_masked, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers"""
        # Use more stable optimizer settings for masked modeling
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Use cosine annealing scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  # Maximum number of iterations
            eta_min=1e-6,  # Minimum learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class SelfSupervisedTrainer:
    """Trainer class for self-supervised learning models"""

    def __init__(self, *args, **kwargs):
        """Initialize trainer - placeholder for compatibility"""
        pass
