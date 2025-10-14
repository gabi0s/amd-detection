"""
AMD OCT Classification - Training Script
Train EfficientNet-B0 for AMD detection
ADAPTÉ POUR STRUCTURE: data/oct2017/OCT2017_/train|test|val
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

class OCTDataset(Dataset):
    """Dataset loader for OCT images"""
    def __init__(self, root_dir, split='train', transform=None):
        base = Path(root_dir)
        
        # Chercher le bon dossier
        possible_paths = [
            base / 'OCT2017_' / split,
            base / 'oct2017' / 'OCT2017_' / split,
        ]
        
        self.root_dir = None
        for path in possible_paths:
            if path.exists():
                self.root_dir = path
                break
        
        if self.root_dir is None:
            raise FileNotFoundError(f"Cannot find {split} folder in {root_dir}")
        
        self.transform = transform
        self.classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        self.samples = []
        
        print(f"Loading {split} dataset from: {self.root_dir}")
        for idx, cls in enumerate(self.classes):
            cls_path = self.root_dir / cls
            if cls_path.exists():
                images = list(cls_path.glob('*.jpeg')) + \
                        list(cls_path.glob('*.jpg')) + \
                        list(cls_path.glob('*.png'))
                for img_path in images:
                    self.samples.append((str(img_path), idx))
                print(f"  {cls}: {len(images)} images")
            else:
                print(f"  WARNING: {cls} folder not found")
        
        print(f"Total loaded: {len(self.samples)} images for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def get_transforms(split='train'):
    """Get data transforms for training/validation"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def create_model(num_classes=4, pretrained=True):
    """Create EfficientNet-B0 model"""
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0()
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(dataloader), 100. * correct / total

def main():
    print("=" * 60)
    print("AMD OCT CLASSIFICATION - FULL TRAINING")
    print("=" * 60)
    
    # Configuration
    DATA_DIR = 'data'
    BATCH_SIZE = 32  # Réduire à 16 si erreur mémoire
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # Create datasets
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    
    train_dataset = OCTDataset(DATA_DIR, split='train', 
                               transform=get_transforms('train'))
    val_dataset = OCTDataset(DATA_DIR, split='test', 
                            transform=get_transforms('val'))
    
    print(f"\n✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    
    model = create_model(num_classes=4, pretrained=True)
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Classes: {train_dataset.classes}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 
               'val_loss': [], 'val_acc': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"\n📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes
            }, 'models/best_model.pth')
            print(f"✓ Saved best model (acc: {val_acc:.2f}%)")
    
    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'models/history_{timestamp}.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"✓ Best validation accuracy: {best_acc:.2f}%")
    print(f"✓ Model saved to: models/best_model.pth")
    print(f"✓ History saved to: models/history_{timestamp}.json")
    print(f"{'='*60}")

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    main()