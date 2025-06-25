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