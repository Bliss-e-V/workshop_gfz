"""
Data utilities for E-OBS climate data handling
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
from pathlib import Path
import xarray as xr
import random


class EOBSDataLoader:
    """Load E-OBS precipitation data from netCDF files"""

    def __init__(self, data_dir: str = "src/data"):
        self.data_dir = Path(data_dir)
        print(f"📊 Initializing EOBSDataLoader with data directory: {self.data_dir}")

    def load_precipitation_data(self):
        """Load precipitation data from E-OBS netCDF files"""
        precipitation_files = {
            "precipitation_mean": self.data_dir / "rr_ens_mean_0.25deg_reg_v31.0e.nc",
            "precipitation_spread": self.data_dir
            / "rr_ens_spread_0.25deg_reg_v31.0e.nc",
        }

        precipitation_data = {}
        for key, file_path in precipitation_files.items():
            if file_path.exists():
                print(f"📊 Loading {key} data from {file_path}")
                dataset = xr.open_dataset(file_path, chunks={"time": 1000})
                print(f"   - Shape: {dataset.dims}")
                print(f"   - Variables: {list(dataset.data_vars.keys())}")
                precipitation_data[key] = dataset
            else:
                print(f"⚠️  File not found: {file_path}")

        return precipitation_data

    def load_elevation_data(self):
        """Load elevation data from E-OBS netCDF files"""
        elevation_file = self.data_dir / "elev_ens_0.25deg_reg_v31.0e.nc"

        if elevation_file.exists():
            print(f"📊 Loading elevation data from {elevation_file}")
            dataset = xr.open_dataset(elevation_file)
            print(f"   - Shape: {dataset.dims}")
            print(f"   - Variables: {list(dataset.data_vars.keys())}")
            return dataset
        else:
            print(f"⚠️  Elevation file not found: {elevation_file}")
            return None

    def load_all_data(self):
        """Load all available E-OBS data"""
        data = {}

        # Load precipitation data
        precip_data = self.load_precipitation_data()
        if precip_data:
            data.update(precip_data)

        # Load elevation data
        elevation_data = self.load_elevation_data()
        if elevation_data is not None:
            data["elevation"] = elevation_data

        return data

    def get_data_info(self, dataset):
        """Get basic information about the dataset"""
        if dataset is None:
            return "No data available"

        info = {
            "dimensions": dict(dataset.dims),
            "variables": list(dataset.data_vars.keys()),
            "coordinates": list(dataset.coords.keys()),
            "time_range": None,
            "spatial_range": None,
        }

        if "time" in dataset.dims:
            time_values = dataset.time.values
            info["time_range"] = (str(time_values[0])[:10], str(time_values[-1])[:10])

        if "latitude" in dataset.dims and "longitude" in dataset.dims:
            info["spatial_range"] = {
                "lat": (float(dataset.latitude.min()), float(dataset.latitude.max())),
                "lon": (float(dataset.longitude.min()), float(dataset.longitude.max())),
            }

        return info


def quick_load_eobs(data_dir: str = "src/data"):
    """Quick function to load E-OBS data"""
    loader = EOBSDataLoader(data_dir)
    return loader.load_all_data()


class EOBSTemporalPredictionDataset(Dataset):
    """Dataset for temporal prediction tasks using E-OBS data"""

    def __init__(
        self,
        precipitation_data: xr.Dataset,
        sequence_length: int = 7,
        prediction_horizon: int = 1,
        variable_name: str = "rr",
        spatial_crop_size: Tuple[int, int] = (64, 64),
        normalize: bool = True,
        log_transform: bool = True,
    ):
        """
        Initialize temporal prediction dataset

        Args:
            precipitation_data: xarray Dataset with precipitation data
            sequence_length: Number of time steps in input sequence
            prediction_horizon: Number of time steps to predict
            variable_name: Name of the precipitation variable
            spatial_crop_size: Size of spatial crops (height, width)
            normalize: Whether to normalize data
            log_transform: Whether to apply log(1+x) transformation
        """
        self.precipitation_data = precipitation_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.variable_name = variable_name
        self.spatial_crop_size = spatial_crop_size
        self.normalize = normalize
        self.log_transform = log_transform

        # Extract precipitation variable
        if variable_name in precipitation_data.data_vars:
            self.precip_var = precipitation_data[variable_name]
        else:
            raise ValueError(f"Variable '{variable_name}' not found in dataset")

        # Calculate valid time indices
        self.num_times = len(self.precip_var.time)
        self.valid_start_indices = list(
            range(self.num_times - self.sequence_length - self.prediction_horizon + 1)
        )

        # Calculate normalization statistics
        if self.normalize:
            self._calculate_stats()

        # Get spatial dimensions
        self.spatial_dims = (
            len(self.precip_var.latitude),
            len(self.precip_var.longitude),
        )

        print("📊 Temporal Dataset created:")
        print(f"   - Sequence length: {self.sequence_length}")
        print(f"   - Prediction horizon: {self.prediction_horizon}")
        print(f"   - Spatial crop size: {self.spatial_crop_size}")
        print(f"   - Total samples: {len(self.valid_start_indices)}")
        print(f"   - Original spatial dims: {self.spatial_dims}")
        print(f"   - Normalize: {self.normalize}")
        print(f"   - Log transform: {self.log_transform}")

    def _calculate_stats(self):
        """Calculate statistics for normalization"""
        print("📊 Calculating normalization statistics...")

        # Sample subset for statistics calculation
        sample_indices = random.sample(
            self.valid_start_indices, min(1000, len(self.valid_start_indices))
        )

        all_values = []
        for idx in sample_indices:
            # Get random spatial crop
            crop_data = self._get_random_spatial_crop(
                self.precip_var.isel(
                    time=slice(
                        idx, idx + self.sequence_length + self.prediction_horizon
                    )
                )
            )

            # Apply preprocessing
            processed_data = self._preprocess(crop_data)
            all_values.append(processed_data.values.flatten())

        all_values = np.concatenate(all_values)

        self.mean = float(np.mean(all_values))
        self.std = float(np.std(all_values))

        print(f"   - Mean: {self.mean:.4f}")
        print(f"   - Std: {self.std:.4f}")

    def _get_random_spatial_crop(self, data):
        """Get random spatial crop from data"""
        height, width = self.spatial_dims
        crop_h, crop_w = self.spatial_crop_size

        # Random crop coordinates
        start_h = random.randint(0, max(0, height - crop_h))
        start_w = random.randint(0, max(0, width - crop_w))

        # Apply crop
        cropped = data.isel(
            latitude=slice(start_h, start_h + crop_h),
            longitude=slice(start_w, start_w + crop_w),
        )

        return cropped

    def _preprocess(self, data):
        """Apply preprocessing to data"""
        # Fill NaN values with 0
        data = data.fillna(0)

        # Apply log transformation if requested
        if self.log_transform:
            data = np.log1p(data)

        return data

    def __len__(self):
        return len(self.valid_start_indices)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        start_idx = self.valid_start_indices[idx]

        # Get time slice
        time_slice = slice(
            start_idx, start_idx + self.sequence_length + self.prediction_horizon
        )
        temporal_data = self.precip_var.isel(time=time_slice)

        # Get spatial crop
        spatial_crop = self._get_random_spatial_crop(temporal_data)

        # Apply preprocessing
        processed_data = self._preprocess(spatial_crop)

        # Convert to numpy and handle any remaining issues
        data_array = processed_data.values

        # Ensure we have the right shape
        if data_array.ndim == 3:  # time, height, width
            pass
        elif data_array.ndim == 2:  # Somehow missing time dimension
            data_array = data_array[np.newaxis, ...]
        else:
            raise ValueError(f"Unexpected data shape: {data_array.shape}")

        # Split into sequence and target
        sequence = data_array[: self.sequence_length]
        target = data_array[
            self.sequence_length : self.sequence_length + self.prediction_horizon
        ]

        # Normalize if requested
        if self.normalize:
            sequence = (sequence - self.mean) / self.std
            target = (target - self.mean) / self.std

        # Convert to torch tensors
        sequence = torch.from_numpy(sequence).float()
        target = torch.from_numpy(target).float()

        # Add channel dimension if needed
        if sequence.dim() == 3:  # time, height, width
            sequence = sequence.unsqueeze(1)  # time, channels, height, width
        if target.dim() == 3:  # time, height, width
            target = target.unsqueeze(1)  # time, channels, height, width

        return sequence, target


class EOBSMaskedModelingDataset(Dataset):
    """Dataset for masked modeling tasks using E-OBS data"""

    def __init__(
        self,
        precipitation_data: xr.Dataset,
        variable_name: str = "rr",
        spatial_size: Tuple[int, int] = (64, 64),
        mask_ratio: float = 0.25,
        mask_strategy: str = "random_patches",  # 'random_patches', 'block', 'irregular'
        patch_size: int = 8,
        normalize: bool = True,
        log_transform: bool = True,
        temporal_context: int = 1,  # Number of time steps to use as context
    ):
        """
        Initialize masked modeling dataset

        Args:
            precipitation_data: xarray Dataset with precipitation data
            variable_name: Name of the precipitation variable
            spatial_size: Size of spatial crops (height, width)
            mask_ratio: Ratio of area to mask (0.0 to 1.0)
            mask_strategy: Strategy for creating masks
            patch_size: Size of patches for masking
            normalize: Whether to normalize data
            log_transform: Whether to apply log(1+x) transformation
            temporal_context: Number of time steps to use
        """
        self.precipitation_data = precipitation_data
        self.variable_name = variable_name
        self.spatial_size = spatial_size
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.patch_size = patch_size
        self.normalize = normalize
        self.log_transform = log_transform
        self.temporal_context = temporal_context

        # Extract precipitation variable
        if variable_name in precipitation_data.data_vars:
            self.precip_var = precipitation_data[variable_name]
        else:
            raise ValueError(f"Variable '{variable_name}' not found in dataset")

        # Calculate valid time indices
        self.num_times = len(self.precip_var.time)
        self.valid_time_indices = list(range(self.num_times))

        # Calculate normalization statistics
        if self.normalize:
            self._calculate_stats()

        # Get spatial dimensions
        self.spatial_dims = (
            len(self.precip_var.latitude),
            len(self.precip_var.longitude),
        )

        print("🎭 Masked Modeling Dataset created:")
        print(f"   - Spatial size: {self.spatial_size}")
        print(f"   - Mask ratio: {self.mask_ratio}")
        print(f"   - Mask strategy: {self.mask_strategy}")
        print(f"   - Patch size: {self.patch_size}")
        print(f"   - Total samples: {len(self.valid_time_indices)}")
        print(f"   - Original spatial dims: {self.spatial_dims}")
        print(f"   - Normalize: {self.normalize}")
        print(f"   - Log transform: {self.log_transform}")

    def _calculate_stats(self):
        """Calculate statistics for normalization"""
        print("📊 Calculating normalization statistics...")

        # Sample subset for statistics calculation
        sample_indices = random.sample(
            self.valid_time_indices, min(1000, len(self.valid_time_indices))
        )

        all_values = []
        for idx in sample_indices:
            # Get random spatial crop
            crop_data = self._get_random_spatial_crop(self.precip_var.isel(time=idx))

            # Apply preprocessing
            processed_data = self._preprocess(crop_data)
            all_values.append(processed_data.values.flatten())

        all_values = np.concatenate(all_values)

        self.mean = float(np.mean(all_values))
        self.std = float(np.std(all_values))

        print(f"   - Mean: {self.mean:.4f}")
        print(f"   - Std: {self.std:.4f}")

    def _get_random_spatial_crop(self, data):
        """Get random spatial crop from data"""
        height, width = self.spatial_dims
        crop_h, crop_w = self.spatial_size

        # Random crop coordinates
        start_h = random.randint(0, max(0, height - crop_h))
        start_w = random.randint(0, max(0, width - crop_w))

        # Apply crop
        cropped = data.isel(
            latitude=slice(start_h, start_h + crop_h),
            longitude=slice(start_w, start_w + crop_w),
        )

        return cropped

    def _create_mask(self, height, width):
        """Create mask for the given spatial dimensions"""
        mask = np.ones((height, width), dtype=np.float32)

        if self.mask_strategy == "random_patches":
            # Random patch masking
            num_patches_h = height // self.patch_size
            num_patches_w = width // self.patch_size
            total_patches = num_patches_h * num_patches_w
            num_masked_patches = int(total_patches * self.mask_ratio)

            # Randomly select patches to mask
            patch_indices = random.sample(range(total_patches), num_masked_patches)

            for patch_idx in patch_indices:
                patch_h = patch_idx // num_patches_w
                patch_w = patch_idx % num_patches_w

                start_h = patch_h * self.patch_size
                end_h = min(start_h + self.patch_size, height)
                start_w = patch_w * self.patch_size
                end_w = min(start_w + self.patch_size, width)

                mask[start_h:end_h, start_w:end_w] = 0.0

        elif self.mask_strategy == "block":
            # Block masking
            mask_h = int(height * np.sqrt(self.mask_ratio))
            mask_w = int(width * np.sqrt(self.mask_ratio))

            start_h = random.randint(0, height - mask_h)
            start_w = random.randint(0, width - mask_w)

            mask[start_h : start_h + mask_h, start_w : start_w + mask_w] = 0.0

        elif self.mask_strategy == "irregular":
            # Irregular masking
            total_pixels = height * width
            num_masked_pixels = int(total_pixels * self.mask_ratio)

            # Create random mask
            flat_mask = np.ones(total_pixels)
            masked_indices = random.sample(range(total_pixels), num_masked_pixels)
            flat_mask[masked_indices] = 0.0

            mask = flat_mask.reshape(height, width)

        return mask

    def _preprocess(self, data):
        """Apply preprocessing to data"""
        # Fill NaN values with 0
        data = data.fillna(0)

        # Apply log transformation if requested
        if self.log_transform:
            data = np.log1p(data)

        return data

    def __len__(self):
        return len(self.valid_time_indices)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        time_idx = self.valid_time_indices[idx]

        # Get time slice
        temporal_data = self.precip_var.isel(time=time_idx)

        # Get spatial crop
        spatial_crop = self._get_random_spatial_crop(temporal_data)

        # Apply preprocessing
        processed_data = self._preprocess(spatial_crop)

        # Convert to numpy
        data_array = processed_data.values

        # Ensure we have 2D spatial data
        if data_array.ndim == 2:
            height, width = data_array.shape
        else:
            raise ValueError(f"Unexpected data shape: {data_array.shape}")

        # Create mask
        mask = self._create_mask(height, width)

        # Apply mask to create input
        masked_input = data_array * mask

        # Normalize if requested
        if self.normalize:
            data_array = (data_array - self.mean) / self.std
            masked_input = (masked_input - self.mean) / self.std

        # Convert to torch tensors
        masked_input = torch.from_numpy(masked_input).float()
        target = torch.from_numpy(data_array).float()
        mask = torch.from_numpy(mask).float()

        # Add channel dimension
        masked_input = masked_input.unsqueeze(0)  # channels, height, width
        target = target.unsqueeze(0)  # channels, height, width

        return masked_input, target, mask


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        print("🎮 Using CUDA GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
        return torch.device("mps")
    else:
        print("💻 Using CPU")
        return torch.device("cpu")
