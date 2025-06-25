"""
Training utilities for the AI Workshop
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import time


class WorkshopTrainer:
    """Simplified trainer for workshop demonstrations"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module,
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 5, lr: float = 0.001) -> Dict[str, List[float]]:
        """Full training loop"""
        
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📊 Model: {self.model.__class__.__name__}")
        print(f"🎯 Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📉 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"🕐 Time: {time.time() - start_time:.1f}s")
        
        print(f"\n✅ Training completed! Total time: {time.time() - start_time:.1f}s")
        return self.history
    
    def quick_train(self, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 3) -> Dict[str, List[float]]:
        """Quick training for workshop demonstrations"""
        print("⚡ Quick training mode (for workshop demo)")
        return self.train(train_loader, val_loader, epochs=epochs, lr=0.001)


def evaluate_model(model: nn.Module, test_loader: DataLoader,
                  device: str = 'cpu') -> Dict[str, float]:
    """Evaluate model and return metrics"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    return {
        'accuracy': accuracy,
        'predictions': np.array(predictions),
        'true_labels': np.array(true_labels)
    }


def compare_models(model1: nn.Module, model2: nn.Module, test_loader: DataLoader,
                  model1_name: str = "Model 1", model2_name: str = "Model 2",
                  device: str = 'cpu') -> Dict:
    """Compare two models side by side"""
    print(f"🔍 Comparing {model1_name} vs {model2_name}")
    
    # Evaluate both models
    results1 = evaluate_model(model1, test_loader, device)
    results2 = evaluate_model(model2, test_loader, device)
    
    # Create comparison
    comparison = {
        model1_name: {
            'accuracy': results1['accuracy'],
            'predictions': results1['predictions'],
            'true_labels': results1['true_labels']
        },
        model2_name: {
            'accuracy': results2['accuracy'],
            'predictions': results2['predictions'],
            'true_labels': results2['true_labels']
        },
        'winner': model1_name if results1['accuracy'] > results2['accuracy'] else model2_name
    }
    
    print(f"📊 {model1_name} Accuracy: {results1['accuracy']:.2f}%")
    print(f"📊 {model2_name} Accuracy: {results2['accuracy']:.2f}%")
    print(f"🏆 Winner: {comparison['winner']}")
    
    return comparison


def create_dummy_history(epochs: int = 5) -> Dict[str, List[float]]:
    """Create dummy training history for quick demonstrations"""
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Simulate improving metrics
        train_loss = 2.0 - epoch * 0.3 + np.random.normal(0, 0.1)
        val_loss = 2.2 - epoch * 0.25 + np.random.normal(0, 0.1)
        train_acc = 20 + epoch * 15 + np.random.normal(0, 2)
        val_acc = 18 + epoch * 12 + np.random.normal(0, 2)
        
        history['train_loss'].append(max(0.1, train_loss))
        history['val_loss'].append(max(0.1, val_loss))
        history['train_acc'].append(min(95, max(10, train_acc)))
        history['val_acc'].append(min(90, max(5, val_acc)))
    
    return history 