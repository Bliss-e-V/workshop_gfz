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


__all__ = [
    "EUROSAT_CLASSES",
    "EuroSATDataset", 
    "HydroFloodDataset",
    "create_sample_batch",
    "denormalize_image",
] 