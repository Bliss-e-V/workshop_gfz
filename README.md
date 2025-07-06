# 🌍 Self-Supervised Learning with E-OBS Climate Data

This repository demonstrates self-supervised learning approaches for E-OBS precipitation data from the Copernicus Climate Change Service.

## 🌦️ Dataset

We work with E-OBS daily precipitation data from the [European Climate Assessment & Dataset project](https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php). The dataset provides:

- **Daily precipitation sums** across Europe
- **0.25° grid resolution** (~25km spatial resolution)
- **Time period**: 1950-2024 (version 31.0e)
- **Spatial coverage**: Europe (-40.4°E to 75.4°E, 25.4°N to 75.4°N)

### 🚀 Quick Start Dataset

For faster development and testing, we provide a **subsampled dataset** with 2000 time steps (1950-1955) that is loaded by default:

- **Subsampled files**: `rr_ens_mean_0.25deg_reg_v31.0e_sub2000.nc_sub` (~27MB each)
- **Time range**: 1950-01-01 to 1955-06-23 (5.5 years)
- **Original files**: `rr_ens_mean_0.25deg_reg_v31.0e.nc` (~348MB each)
- **Automatic fallback**: If subsampled files are not found, the full dataset is loaded

> 💡 **Note**: The data loader automatically prefers the subsampled files for faster loading and experimentation. To force loading the full dataset, use `loader.load_precipitation_data(prefer_subsampled=False)`.

## 🎯 Self-Supervised Learning Approaches

### 1. Temporal Prediction 🔮
- **Task**: Predict next day's precipitation from past 3 days
- **Architecture**: Transformer-based model with patch embeddings
- **Learning**: Temporal patterns and weather evolution
- **Use case**: Weather forecasting, temporal pattern learning

### 2. Masked Modeling 🎭
- **Task**: Reconstruct masked spatial regions of precipitation maps
- **Architecture**: U-Net encoder-decoder with skip connections
- **Masking strategies**: Random patches, blocks, irregular patterns
- **Use case**: Data imputation, spatial relationship learning

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Example

Open and run the main notebook:

```bash
jupyter notebook notebooks/self_supervised_eobs_example.ipynb
```

This notebook demonstrates:
- Loading and exploring E-OBS precipitation data
- Creating temporal prediction datasets
- Training temporal prediction models
- Creating masked modeling datasets  
- Training masked modeling models
- Evaluating and visualizing results

## 📁 Repository Structure

```
hydrology_seminar/
├── notebooks/
│   └── self_supervised_eobs_example.ipynb    # Main demonstration notebook
├── src/
|   ├── data/
|   |   ├── rr_ens_mean_0.25deg_reg_v31.0e_sub2000.nc      # Subsampled precipitation mean (2000 time steps)
|   |   ├── rr_ens_spread_0.25deg_reg_v31.0e_sub2000.nc    # Subsampled precipitation spread (2000 time steps)
|   |   ├── rr_ens_mean_0.25deg_reg_v31.0e.nc              # Full precipitation mean dataset
|   |   ├── rr_ens_spread_0.25deg_reg_v31.0e.nc            # Full precipitation spread dataset
|   |   └── elev_ens_0.25deg_reg_v31.0e.nc                 # Elevation data
│   ├── data_utils.py                         # E-OBS data loading and processing
│   └── models.py                             # Self-supervised learning models
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

## 📊 Models and Architectures

### TemporalPredictionModel
- **Transformer-based** with patch embeddings for spatial processing
- **Multi-layer architecture** with self-attention
- **Temporal modeling** for sequence prediction
- **Lightweight design** optimized for climate data

### MaskedModelingModel
- **U-Net architecture** with encoder-decoder structure
- **Skip connections** for spatial detail preservation
- **Flexible masking strategies** for robust learning
- **Batch normalization** for stable training

## 🔧 Key Components

### Data Loading (`EOBSDataLoader`)
- Loads E-OBS netCDF files with chunking for memory efficiency
- Handles precipitation mean and spread data
- Provides data information and statistics

### Dataset Classes
- `EOBSTemporalPredictionDataset`: Creates temporal sequences for forecasting
- `EOBSMaskedModelingDataset`: Creates masked maps for reconstruction

### Lightning Modules
- `TemporalPredictionLightningModule`: PyTorch Lightning wrapper for temporal models
- `MaskedModelingLightningModule`: PyTorch Lightning wrapper for masked models

## 🎓 Learning Outcomes

By exploring this repository, you will:
1. Understand self-supervised learning principles for climate data
2. Learn to work with large-scale precipitation datasets
3. Implement transformer-based temporal prediction models
4. Build U-Net architectures for spatial reconstruction
5. Use PyTorch Lightning for efficient training workflows
6. Evaluate model performance on real climate data

## 💡 Key Insights

Self-supervised learning with weather data presents unique challenges:
- **Weather chaos**: Small changes can lead to large differences
- **Non-linear dynamics**: Precipitation patterns are complex
- **Multi-scale patterns**: From local to synoptic scales
- **Missing context**: Need temperature, pressure, humidity for full picture

This repository demonstrates both the potential and limitations of self-supervised learning for climate data.

## 🔗 References

- [E-OBS Dataset](https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php)
- [Copernicus Climate Change Service](https://climate.copernicus.eu/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

---

*Self-supervised learning for climate science - bridging AI and atmospheric physics* 🌍 