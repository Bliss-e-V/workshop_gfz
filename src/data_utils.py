"""
Data utilities for the AI Workshop - EuroSAT dataset handling
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict

# EuroSAT class names
EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]


class EuroSATDataset:
    """Helper class to handle EuroSAT dataset operations"""
    
    def __init__(self, root_dir: str = './data', download: bool = True):
        self.root_dir = root_dir
        self.classes = EUROSAT_CLASSES
        self.num_classes = len(self.classes)
        
        # Define transforms
        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for compatibility
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        self._load_dataset(download)
    
    def _load_dataset(self, download: bool):
        """Load EuroSAT dataset - we'll use a simple approach for the workshop"""
        # For workshop purposes, we'll simulate loading the EuroSAT dataset
        # In a real scenario, you'd download from the official source
        
        # Create dummy data for demonstration (replace with actual dataset loading)
        print("🛰️  Loading EuroSAT dataset...")
        print("📊  Dataset Info:")
        print(f"   - Classes: {self.num_classes}")
        print(f"   - Class names: {', '.join(self.classes[:5])}...")
        print("   - Image size: 64x64 -> 224x224 (resized)")
        print("   - Total samples: ~27,000")
        
    def get_dataloaders(self, batch_size: int = 32, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Get training and validation dataloaders"""
        # For workshop demo, we'll use CIFAR-10 as a substitute
        # In practice, replace this with actual EuroSAT loading
        
        dataset_train = torchvision.datasets.CIFAR10(
            root=self.root_dir, train=True, download=True, 
            transform=self.transform_train
        )
        dataset_val = torchvision.datasets.CIFAR10(
            root=self.root_dir, train=False, download=True,
            transform=self.transform_val
        )
        
        # Map CIFAR-10 to our classes (for demo purposes)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def visualize_samples(self, dataloader: DataLoader, num_samples: int = 8):
        """Display sample images from the dataset"""
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('🛰️ EuroSAT Dataset Samples', fontsize=16, fontweight='bold')
        
        for i in range(num_samples):
            row, col = i // 4, i % 4
            
            # Denormalize image for display
            img = images[i].clone()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            axes[row, col].imshow(img.permute(1, 2, 0))
            axes[row, col].set_title(f'{self.classes[labels[i] % self.num_classes]}', 
                                   fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self, dataloader: DataLoader) -> Dict[str, int]:
        """Get class distribution statistics"""
        class_counts = {cls: 0 for cls in self.classes}
        
        for _, labels in dataloader:
            for label in labels:
                class_counts[self.classes[label % self.num_classes]] += 1
        
        return class_counts
    
    def plot_class_distribution(self, class_counts: Dict[str, int]):
        """Plot class distribution"""
        plt.figure(figsize=(12, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        bars = plt.bar(classes, counts, color=colors)
        
        plt.title('📊 EuroSAT Dataset - Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Land Use Classes', fontweight='bold')
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()


def create_sample_batch(batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a sample batch for quick testing"""
    # Create random images (3, 224, 224) and labels
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, len(EUROSAT_CLASSES), (batch_size,))
    return images, labels


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


# --- Hydrology-specific dataset -------------------------------------------------


class HydroFloodDataset:
    """Tiny flood-mapping set (Sentinel-1/2) – 2 classes: water / no-water."""
    LABELS = ["no-water", "water"]

    def __init__(
        self,
        root_dir: str = "./data",
        split: str = "train",
        transform: transforms.Compose | None = None,
        download: bool = True,
    ):
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        try:
            from datasets import load_dataset  # lazy import
            ds = load_dataset("see/floodnet-mini", split=split)
            self.images, self.labels = ds["image"], ds["label"]
            print(f"🌊 Flood dataset – {len(self.images)} samples ({split}).")
        except Exception as err:
            print(f"⚠️ Flood dataset unavailable ({err}); falling back to CIFAR-10 proxy.")
            cifar = torchvision.datasets.CIFAR10(root=root_dir,
                                                 train=(split == "train"),
                                                 download=download)
            self.images = [img for img, _ in cifar]
            self.labels = [1 if lbl == 8 else 0 for _, lbl in cifar]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.transform(self.images[idx])
        return img, int(self.labels[idx])

    def get_dataloaders(self, batch_size=32, val_split=0.2, shuffle=True):
        total = len(self)
        val_size = int(total * val_split)
        train_ds, val_ds = torch.utils.data.random_split(
            self, [total - val_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        dl_args = dict(batch_size=batch_size)
        return (
            DataLoader(train_ds, shuffle=shuffle, **dl_args),
            DataLoader(val_ds, shuffle=False, **dl_args),
        )


# -----------------------------------------------------------------------------


# --- E-OBS Climate Data Loading Functions ---------------------------------------

import xarray as xr
from pathlib import Path


class EOBSDataLoader:
    """
    Loader for E-OBS climate dataset from Copernicus Climate Change Service
    https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php
    """
    
    def __init__(self, data_dir: str = "src/data"):
        self.data_dir = Path(data_dir)
        
    def load_precipitation_data(self):
        """Load precipitation mean and spread data"""
        mean_file = self.data_dir / "rr_ens_mean_0.25deg_reg_v31.0e.nc"
        spread_file = self.data_dir / "rr_ens_spread_0.25deg_reg_v31.0e.nc"
        
        data = {}
        
        if mean_file.exists():
            print(f"📊 Loading precipitation mean data from {mean_file}")
            data['precipitation_mean'] = xr.open_dataset(mean_file)
            print(f"   - Shape: {data['precipitation_mean'].dims}")
            print(f"   - Variables: {list(data['precipitation_mean'].data_vars.keys())}")
        else:
            print(f"⚠️  Precipitation mean file not found: {mean_file}")
            
        if spread_file.exists():
            print(f"📊 Loading precipitation spread data from {spread_file}")
            data['precipitation_spread'] = xr.open_dataset(spread_file)
            print(f"   - Shape: {data['precipitation_spread'].dims}")
            print(f"   - Variables: {list(data['precipitation_spread'].data_vars.keys())}")
        else:
            print(f"⚠️  Precipitation spread file not found: {spread_file}")
            
        return data
    
    def load_elevation_data(self):
        """Load elevation ensemble data"""
        elev_file = self.data_dir / "elev_ens_0.25deg_reg_v31.0e.nc"
        
        if elev_file.exists():
            print(f"🏔️  Loading elevation data from {elev_file}")
            elevation_data = xr.open_dataset(elev_file)
            print(f"   - Shape: {elevation_data.dims}")
            print(f"   - Variables: {list(elevation_data.data_vars.keys())}")
            return elevation_data
        else:
            print(f"⚠️  Elevation file not found: {elev_file}")
            return None
    
    def load_all_data(self):
        """Load all available E-OBS data"""
        print("🌍 Loading E-OBS Climate Dataset...")
        
        data = {}
        
        # Load precipitation data
        precip_data = self.load_precipitation_data()
        data.update(precip_data)
        
        # Load elevation data
        elevation_data = self.load_elevation_data()
        if elevation_data is not None:
            data['elevation'] = elevation_data
            
        print(f"\n✅ Loaded {len(data)} datasets")
        return data
    
    def get_data_info(self, dataset):
        """Get comprehensive information about a dataset"""
        if not isinstance(dataset, xr.Dataset):
            print("❌ Please provide an xarray Dataset")
            return
            
        print(f"\n📋 Dataset Information:")
        print(f"   - Dimensions: {dict(dataset.dims)}")
        print(f"   - Coordinates: {list(dataset.coords.keys())}")
        print(f"   - Data variables: {list(dataset.data_vars.keys())}")
        
        # Print variable details
        for var_name, var in dataset.data_vars.items():
            print(f"\n🔍 Variable: {var_name}")
            print(f"   - Shape: {var.shape}")
            print(f"   - Dtype: {var.dtype}")
            if 'units' in var.attrs:
                print(f"   - Units: {var.attrs['units']}")
            if 'long_name' in var.attrs:
                print(f"   - Description: {var.attrs['long_name']}")
    
    def plot_sample_data(self, dataset, variable_name=None, time_index=0):
        """Plot a sample of the data"""
        import matplotlib.pyplot as plt
        
        if variable_name is None:
            variable_name = list(dataset.data_vars.keys())[0]
            
        if variable_name not in dataset.data_vars:
            print(f"❌ Variable '{variable_name}' not found in dataset")
            print(f"Available variables: {list(dataset.data_vars.keys())}")
            return
            
        data_var = dataset[variable_name]
        
        # Handle different dimensionalities
        if 'time' in data_var.dims:
            if len(data_var.dims) == 3:  # time, lat, lon
                plot_data = data_var.isel(time=time_index)
            else:
                print(f"⚠️  Unsupported dimensions: {data_var.dims}")
                return
        else:
            plot_data = data_var
            
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = plot_data.plot(ax=ax, cmap='viridis', add_colorbar=True)
        
        ax.set_title(f'E-OBS {variable_name.title()} Data', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
        
        plt.tight_layout()
        plt.show()


def quick_load_eobs(data_dir: str = "src/data"):
    """Quick function to load E-OBS data"""
    loader = EOBSDataLoader(data_dir)
    return loader.load_all_data()


# --- Self-Supervised Learning Datasets for E-OBS -------------------------------

import random
from torch.utils.data import Dataset


class EOBSTemporalPredictionDataset(Dataset):
    """
    Self-supervised dataset for temporal prediction of precipitation.
    Uses past N days to predict the next day's precipitation.
    """
    
    def __init__(
        self,
        precipitation_data: xr.Dataset,
        sequence_length: int = 7,
        prediction_horizon: int = 1,
        variable_name: str = 'rr',
        spatial_crop_size: Tuple[int, int] = (64, 64),
        normalize: bool = True,
        log_transform: bool = True
    ):
        """
        Args:
            precipitation_data: xarray Dataset with precipitation data
            sequence_length: Number of past days to use as input (default: 7)
            prediction_horizon: Number of days ahead to predict (default: 1)
            variable_name: Name of precipitation variable in dataset
            spatial_crop_size: Size of spatial crops (H, W)
            normalize: Whether to normalize data
            log_transform: Whether to apply log(1+x) transform to precipitation
        """
        self.data = precipitation_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.variable_name = variable_name
        self.spatial_crop_size = spatial_crop_size
        self.normalize = normalize
        self.log_transform = log_transform
        
        # Get the precipitation variable
        if variable_name not in self.data.data_vars:
            available_vars = list(self.data.data_vars.keys())
            raise ValueError(f"Variable '{variable_name}' not found. Available: {available_vars}")
        
        self.precip_data = self.data[variable_name]
        
        # Get dimensions
        self.time_dim = len(self.precip_data.time)
        self.lat_dim = len(self.precip_data.latitude)
        self.lon_dim = len(self.precip_data.longitude)
        
        # Valid time indices for sequences
        self.valid_time_indices = list(range(
            self.sequence_length, 
            self.time_dim - self.prediction_horizon + 1
        ))
        
        # Calculate normalization statistics if needed
        if self.normalize:
            self._calculate_stats()
        
        print(f"🔮 Temporal Prediction Dataset initialized:")
        print(f"   - Total time steps: {self.time_dim}")
        print(f"   - Valid sequences: {len(self.valid_time_indices)}")
        print(f"   - Sequence length: {self.sequence_length}")
        print(f"   - Prediction horizon: {self.prediction_horizon}")
        print(f"   - Spatial size: {self.lat_dim} x {self.lon_dim}")
        print(f"   - Crop size: {self.spatial_crop_size}")
    
    def _calculate_stats(self):
        """Calculate mean and std for normalization"""
        print("📊 Calculating normalization statistics...")
        
        # Sample a subset for efficiency
        sample_indices = np.random.choice(
            len(self.precip_data.time), 
            min(1000, len(self.precip_data.time)), 
            replace=False
        )
        sample_data = self.precip_data.isel(time=sample_indices)
        
        # Handle NaN values in sample data
        sample_data = np.nan_to_num(sample_data, nan=0.0, posinf=0.0, neginf=0.0)
        sample_data = np.maximum(sample_data, 0.0)
        
        if self.log_transform:
            sample_data = np.log1p(sample_data)
        
        self.mean = float(np.mean(sample_data))
        self.std = float(np.std(sample_data))
        
        print(f"   - Mean: {self.mean:.4f}")
        print(f"   - Std: {self.std:.4f}")
    
    def _get_random_spatial_crop(self, data):
        """Get a random spatial crop from the data"""
        h, w = self.spatial_crop_size
        max_lat_idx = self.lat_dim - h
        max_lon_idx = self.lon_dim - w
        
        if max_lat_idx <= 0 or max_lon_idx <= 0:
            # If crop size is larger than data, return full data
            return data
        
        lat_start = random.randint(0, max_lat_idx)
        lon_start = random.randint(0, max_lon_idx)
        
        return data[..., lat_start:lat_start+h, lon_start:lon_start+w]
    
    def _preprocess(self, data):
        """Apply preprocessing transformations"""
        # Handle NaN values by replacing with 0 (no precipitation)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure non-negative values for log transform
        data = np.maximum(data, 0.0)
        
        if self.log_transform:
            data = np.log1p(data)
        
        if self.normalize:
            data = (data - self.mean) / (self.std + 1e-8)
        
        return data
    
    def __len__(self):
        return len(self.valid_time_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_sequence: Tensor of shape (sequence_length, H, W)
            target: Tensor of shape (prediction_horizon, H, W)
        """
        time_idx = self.valid_time_indices[idx]
        
        # Get input sequence (past days)
        input_indices = list(range(
            time_idx - self.sequence_length, 
            time_idx
        ))
        input_data = self.precip_data.isel(time=input_indices)
        
        # Get target (future days)
        target_indices = list(range(
            time_idx, 
            time_idx + self.prediction_horizon
        ))
        target_data = self.precip_data.isel(time=target_indices)
        
        # Convert to numpy
        input_seq = input_data.values  # (seq_len, lat, lon)
        target = target_data.values    # (pred_horizon, lat, lon)
        
        # Apply the SAME random spatial crop to both input and target
        # Generate crop coordinates once
        h, w = self.spatial_crop_size
        max_lat_idx = self.lat_dim - h
        max_lon_idx = self.lon_dim - w
        
        if max_lat_idx > 0 and max_lon_idx > 0:
            lat_start = random.randint(0, max_lat_idx)
            lon_start = random.randint(0, max_lon_idx)
            
            # Apply same crop to both input and target
            input_seq = input_seq[..., lat_start:lat_start + h, lon_start:lon_start + w]
            target = target[..., lat_start:lat_start + h, lon_start:lon_start + w]
        
        # Preprocess
        input_seq = self._preprocess(input_seq)
        target = self._preprocess(target)
        
        # Convert to torch tensors
        input_seq = torch.FloatTensor(input_seq)
        target = torch.FloatTensor(target)
        
        return input_seq, target


class EOBSMaskedModelingDataset(Dataset):
    """
    Self-supervised dataset for masked modeling of precipitation.
    Randomly masks spatial regions and reconstructs them.
    """
    
    def __init__(
        self,
        precipitation_data: xr.Dataset,
        variable_name: str = 'rr',
        spatial_size: Tuple[int, int] = (64, 64),
        mask_ratio: float = 0.25,
        mask_strategy: str = 'random_patches',  # 'random_patches', 'block', 'irregular'
        patch_size: int = 8,
        normalize: bool = True,
        log_transform: bool = True,
        temporal_context: int = 1  # Number of time steps to use as context
    ):
        """
        Args:
            precipitation_data: xarray Dataset with precipitation data
            variable_name: Name of precipitation variable in dataset
            spatial_size: Size of spatial patches (H, W)
            mask_ratio: Fraction of spatial area to mask (0-1)
            mask_strategy: Strategy for masking ('random_patches', 'block', 'irregular')
            patch_size: Size of patches for patch-based masking
            normalize: Whether to normalize data
            log_transform: Whether to apply log(1+x) transform
            temporal_context: Number of time steps to include (1 = single day)
        """
        self.data = precipitation_data
        self.variable_name = variable_name
        self.spatial_size = spatial_size
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.patch_size = patch_size
        self.normalize = normalize
        self.log_transform = log_transform
        self.temporal_context = temporal_context
        
        # Get the precipitation variable
        if variable_name not in self.data.data_vars:
            available_vars = list(self.data.data_vars.keys())
            raise ValueError(f"Variable '{variable_name}' not found. Available: {available_vars}")
        
        self.precip_data = self.data[variable_name]
        
        # Get dimensions
        self.time_dim = len(self.precip_data.time)
        self.lat_dim = len(self.precip_data.latitude)
        self.lon_dim = len(self.precip_data.longitude)
        
        # Valid time indices
        if temporal_context > 1:
            self.valid_time_indices = list(range(
                temporal_context - 1, 
                self.time_dim
            ))
        else:
            self.valid_time_indices = list(range(self.time_dim))
        
        # Calculate normalization statistics if needed
        if self.normalize:
            self._calculate_stats()
        
        print(f"🎭 Masked Modeling Dataset initialized:")
        print(f"   - Total time steps: {self.time_dim}")
        print(f"   - Valid samples: {len(self.valid_time_indices)}")
        print(f"   - Spatial size: {self.lat_dim} x {self.lon_dim}")
        print(f"   - Target size: {self.spatial_size}")
        print(f"   - Mask ratio: {self.mask_ratio}")
        print(f"   - Mask strategy: {self.mask_strategy}")
        print(f"   - Temporal context: {self.temporal_context}")
    
    def _calculate_stats(self):
        """Calculate mean and std for normalization"""
        print("📊 Calculating normalization statistics...")
        
        # Sample a subset for efficiency
        sample_indices = np.random.choice(
            len(self.precip_data.time), 
            min(1000, len(self.precip_data.time)), 
            replace=False
        )
        sample_data = self.precip_data.isel(time=sample_indices)
        
        # Handle NaN values in sample data
        sample_data = np.nan_to_num(sample_data, nan=0.0, posinf=0.0, neginf=0.0)
        sample_data = np.maximum(sample_data, 0.0)
        
        if self.log_transform:
            sample_data = np.log1p(sample_data)
        
        self.mean = float(np.mean(sample_data))
        self.std = float(np.std(sample_data))
        
        print(f"   - Mean: {self.mean:.4f}")
        print(f"   - Std: {self.std:.4f}")
    
    def _get_random_spatial_crop(self, data):
        """Get a random spatial crop from the data"""
        h, w = self.spatial_size
        
        if len(data.shape) == 3:  # (time, lat, lon)
            max_lat_idx = data.shape[1] - h
            max_lon_idx = data.shape[2] - w
        else:  # (lat, lon)
            max_lat_idx = data.shape[0] - h
            max_lon_idx = data.shape[1] - w
        
        if max_lat_idx <= 0 or max_lon_idx <= 0:
            return data
        
        lat_start = random.randint(0, max_lat_idx)
        lon_start = random.randint(0, max_lon_idx)
        
        if len(data.shape) == 3:
            return data[:, lat_start:lat_start+h, lon_start:lon_start+w]
        else:
            return data[lat_start:lat_start+h, lon_start:lon_start+w]
    
    def _create_mask(self, height, width):
        """Create a binary mask based on the specified strategy"""
        mask = np.zeros((height, width), dtype=bool)
        
        total_pixels = height * width
        target_masked = int(total_pixels * self.mask_ratio)
        
        if self.mask_strategy == 'random_patches':
            # Random patch-based masking
            num_patches_h = height // self.patch_size
            num_patches_w = width // self.patch_size
            total_patches = num_patches_h * num_patches_w
            num_masked_patches = int(total_patches * self.mask_ratio)
            
            # Randomly select patches to mask
            patch_indices = random.sample(range(total_patches), num_masked_patches)
            
            for patch_idx in patch_indices:
                patch_row = patch_idx // num_patches_w
                patch_col = patch_idx % num_patches_w
                
                start_h = patch_row * self.patch_size
                end_h = min(start_h + self.patch_size, height)
                start_w = patch_col * self.patch_size
                end_w = min(start_w + self.patch_size, width)
                
                mask[start_h:end_h, start_w:end_w] = True
        
        elif self.mask_strategy == 'block':
            # Single large block masking
            block_h = int(height * np.sqrt(self.mask_ratio))
            block_w = int(width * np.sqrt(self.mask_ratio))
            
            start_h = random.randint(0, height - block_h)
            start_w = random.randint(0, width - block_w)
            
            mask[start_h:start_h+block_h, start_w:start_w+block_w] = True
        
        elif self.mask_strategy == 'irregular':
            # Random pixel masking
            masked_pixels = random.sample(range(total_pixels), target_masked)
            for pixel_idx in masked_pixels:
                row = pixel_idx // width
                col = pixel_idx % width
                mask[row, col] = True
        
        return mask
    
    def _preprocess(self, data):
        """Apply preprocessing transformations"""
        # Handle NaN values by replacing with 0 (no precipitation)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure non-negative values for log transform
        data = np.maximum(data, 0.0)
        
        if self.log_transform:
            data = np.log1p(data)
        
        if self.normalize:
            data = (data - self.mean) / (self.std + 1e-8)
        
        return data
    
    def __len__(self):
        return len(self.valid_time_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            masked_input: Tensor with masked regions (temporal_context, H, W)
            target: Original tensor (temporal_context, H, W)  
            mask: Binary mask indicating masked regions (H, W)
        """
        time_idx = self.valid_time_indices[idx]
        
        # Get temporal context
        if self.temporal_context > 1:
            context_indices = list(range(
                time_idx - self.temporal_context + 1, 
                time_idx + 1
            ))
            data = self.precip_data.isel(time=context_indices)
        else:
            data = self.precip_data.isel(time=[time_idx])
        
        # Convert to numpy and crop
        data_array = data.values  # (temporal_context, lat, lon)
        if data_array.ndim == 2:
            data_array = data_array[np.newaxis, ...]  # Add time dimension
        
        data_cropped = self._get_random_spatial_crop(data_array)
        
        # Preprocess
        data_processed = self._preprocess(data_cropped)
        
        # Create mask
        _, h, w = data_processed.shape
        mask = self._create_mask(h, w)
        
        # Apply mask to create input
        masked_input = data_processed.copy()
        masked_input[:, mask] = 0  # Zero out masked regions
        
        # Convert to torch tensors
        masked_input = torch.FloatTensor(masked_input)
        target = torch.FloatTensor(data_processed)
        mask_tensor = torch.BoolTensor(mask)
        
        return masked_input, target, mask_tensor


def get_device():
    """
    Automatically detect and return the best available device.
    Prioritizes: MPS (Apple Silicon) > CUDA > CPU
    
    Returns:
        str: Device string ('mps', 'cuda', or 'cpu')
    """
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("💻 Using CPU (consider GPU for faster training)")
    
    return device


__all__ = [
    "EUROSAT_CLASSES",
    "EuroSATDataset", 
    "HydroFloodDataset",
    "create_sample_batch",
    "denormalize_image",
    "EOBSDataLoader",
    "EOBSTemporalPredictionDataset",
    "EOBSMaskedModelingDataset",
    "quick_load_eobs",
    "get_device",
]