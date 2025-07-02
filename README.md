# 🚀 Introduction to AI Workshop (90 minutes)
## Satellite Imagery Classification: CNN vs Transformers

### 🎯 Workshop Overview
This workshop provides hands-on experience with modern AI techniques using **satellite imagery classification**. We'll explore the EuroSAT dataset to classify different land use types from satellite images.

### 📚 What You'll Learn
- **CNN Architecture**: Convolutional Neural Networks for image classification
- **Vision Transformers**: Modern transformer-based computer vision
- **Feature Visualization**: CNN activation maps and transformer attention
- **HuggingFace Integration**: Using pre-trained models
- **Model Comparison**: Performance analysis and interpretability

### 🛰️ Dataset: EuroSAT
- **10 land use classes**: Industrial, Residential, AnnualCrop, Forest, etc.
- **27,000 labeled images** from Sentinel-2 satellite
- **64x64 RGB images** - perfect for workshop timing
- **Real-world application**: Land monitoring and urban planning
- **Optional Sentinel-1/2 flood subset** (binary water mask) to align with GFZ 'Flood Risk & Climate Adaptation' topic

### 📋 Workshop Structure (90 minutes)

#### Part 1: Data Exploration & Setup (15 min)
- Dataset introduction and exploration
- Visualization of satellite imagery classes
- Data preprocessing pipeline

#### Part 2: CNN Implementation (25 min)
- Build a custom CNN from scratch
- Training and evaluation
- **CNN Feature Maps Visualization**
- Performance analysis

#### Part 3: Vision Transformer (25 min)
- Introduction to Vision Transformers (ViT)
- HuggingFace pre-trained ViT model
- **Attention Score Visualization**
- Fine-tuning techniques

#### Part 4: Model Comparison & Advanced Topics (20 min)
- Side-by-side performance comparison
- Transfer learning examples
- Ensemble methods
- Real-world deployment considerations

#### Part 5: Hands-on Exercises (5 min)
- Quick challenges for participants

### 🔧 Setup Instructions

#### Prerequisites
```bash
# Create virtual environment
python -m venv ai_workshop
source ai_workshop/bin/activate  # On Windows: ai_workshop\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Quick Start
```bash
# Launch the main workshop notebook
jupyter notebook 01_workshop_main.ipynb
```

### 📁 File Structure
```
├── notebooks/
│   ├── 01_workshop_main.ipynb          # Main workshop notebook
│   ├── 02_cnn_deep_dive.ipynb          # CNN implementation details
│   ├── 03_transformer_exploration.ipynb # Transformer deep dive
│   └── 04_advanced_topics.ipynb        # Additional exercises
├── src/
│   ├── data_utils.py                   # Data loading and preprocessing
│   ├── models.py                       # Model architectures
│   ├── visualization.py               # Plotting and visualization
│   └── training.py                     # Training utilities
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

### 🎯 Learning Outcomes
By the end of this workshop, participants will:
1. Understand the differences between CNNs and Vision Transformers
2. Know how to visualize and interpret model decisions
3. Be able to use HuggingFace for computer vision tasks
4. Have hands-on experience with real satellite imagery data
5. Understand practical considerations for model deployment

### 🚀 Next Steps
- Explore larger datasets (ImageNet, COCO)
- Try other transformer architectures (DETR, CLIP)
- Implement object detection and segmentation
- Build production pipelines with MLOps tools

---
*Workshop Duration: 90 minutes | Difficulty: Beginner to Intermediate* 

# Hydrology Seminar - Self-Supervised Learning with E-OBS Climate Data

This repository contains implementations of self-supervised learning approaches for E-OBS precipitation data from the Copernicus Climate Change Service.

## 🌍 Dataset

We work with E-OBS daily precipitation data from the [European Climate Assessment & Dataset project](https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php). The dataset provides:

- **Daily precipitation sums** across Europe
- **0.25° grid resolution** 
- **Time period**: 1950-2024 (version 31.0e)
- **Spatial coverage**: Europe

## 🎯 Self-Supervised Learning Approaches

### 1. Temporal Prediction 🔮
- **Task**: Predict next day's precipitation from past 7 days
- **Architecture**: ConvLSTM-based model for spatio-temporal forecasting
- **Use case**: Weather prediction, temporal pattern learning

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

### Basic Usage

```python
from src.data_utils import EOBSDataLoader, EOBSTemporalPredictionDataset, get_device
from src.models import TemporalPredictionModel, SelfSupervisedTrainer

# Auto-detect best device (MPS, CUDA, or CPU)
device = get_device()  # 🍎 Will use MPS on Apple Silicon

# Load E-OBS data
loader = EOBSDataLoader(data_dir="src/data")
eobs_data = loader.load_all_data()
precip_data = eobs_data['precipitation_mean']

# Create temporal prediction dataset
dataset = EOBSTemporalPredictionDataset(
    precipitation_data=precip_data,
    sequence_length=7,
    prediction_horizon=1,
    spatial_crop_size=(64, 64)
)

# Create model and trainer
model = TemporalPredictionModel(
    input_channels=1,
    hidden_channels=64,
    sequence_length=7
)

trainer = SelfSupervisedTrainer(
    model=model,
    task_type='temporal_prediction'
    # device will be auto-detected (MPS/CUDA/CPU)
)
```

### Example Notebooks

- `notebooks/self_supervised_eobs_example.ipynb` - Complete demonstration of both approaches
- `notebooks/01_workshop_main.ipynb` - Original workshop content

## 📊 Dataset Classes

### EOBSTemporalPredictionDataset
Creates temporal sequences for next-day precipitation prediction.

**Key Parameters:**
- `sequence_length`: Number of past days (default: 7)
- `prediction_horizon`: Days to predict ahead (default: 1)
- `spatial_crop_size`: Size of spatial patches (default: 64×64)
- `log_transform`: Apply log(1+x) to precipitation (default: True)

### EOBSMaskedModelingDataset
Creates masked precipitation maps for spatial reconstruction.

**Key Parameters:**
- `mask_ratio`: Fraction of area to mask (default: 0.25)
- `mask_strategy`: 'random_patches', 'block', or 'irregular'
- `patch_size`: Size of masking patches (default: 8)
- `temporal_context`: Number of time steps (default: 1)

## 🏗️ Model Architectures

### TemporalPredictionModel
- **ConvLSTM layers** for spatio-temporal processing
- **Multi-layer architecture** with configurable depth
- **Output projection** for precipitation prediction

### MaskedModelingModel
- **U-Net encoder-decoder** with skip connections
- **Multi-scale processing** through downsampling/upsampling
- **Spatial reconstruction** from masked inputs

### SelfSupervisedTrainer
- **Unified training interface** for both tasks
- **Early stopping** and model checkpointing
- **Training curve visualization**
- **Task-specific loss functions**

## 📈 Applications

### 1. Weather Forecasting
- Short-term precipitation prediction
- Extreme weather event detection
- Seasonal pattern forecasting

### 2. Data Imputation
- Fill missing precipitation data
- Reconstruct damaged satellite observations
- Generate high-resolution data from coarse inputs

### 3. Climate Analysis
- Extract meaningful climate representations
- Transfer learning for downstream tasks
- Regional climate pattern discovery

## 🔬 Advanced Features

### Physics-Informed Learning
- Conservation law constraints
- Topography-aware modeling
- Multi-variable relationships

### Multi-Scale Modeling
- Hierarchical temporal patterns (daily → seasonal)
- Spatial scale relationships
- Cross-resolution prediction

### Contrastive Learning
- Meteorological similarity learning
- Spatio-temporal coherence
- Weather pattern classification

## 📂 Project Structure

```
hydrology_seminar/
├── src/
│   ├── data_utils.py          # E-OBS data loading and SSL datasets
│   ├── models.py              # Neural network architectures
│   ├── training.py            # Training utilities
│   ├── visualization.py       # Plotting functions
│   └── data/                  # E-OBS data files
├── notebooks/
│   ├── self_supervised_eobs_example.ipynb
│   └── 01_workshop_main.ipynb
├── requirements.txt
└── README.md
```

## 🛠️ Requirements

- Python 3.8+
- PyTorch 1.9+
- xarray
- matplotlib
- numpy
- netCDF4

### GPU Acceleration
- **Apple Silicon (M1/M2/M3)**: Automatic MPS (Metal Performance Shaders) detection 🍎
- **NVIDIA GPUs**: CUDA support for faster training 🚀
- **CPU**: Fallback option (slower but functional) 💻

## 📚 References

- [E-OBS Dataset](https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php)
- Cornes, R., et al. (2018). An Ensemble Version of the E-OBS Temperature and Precipitation Datasets. *Journal of Geophysical Research: Atmospheres*.

## 🤝 Contributing

Feel free to contribute by:
- Adding new self-supervised learning approaches
- Improving model architectures  
- Adding evaluation metrics
- Creating visualization tools

## 📄 License

This project is for educational and research purposes. 