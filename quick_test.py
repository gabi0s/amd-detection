"""
AMD OCT Classification - Quick Test (5 min max)
Test rapide avec subset du dataset pour valider le pipeline
ADAPTÉ POUR STRUCTURE: data/oct2017/OCT2017_/train|test|val
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import random

class OCTDataset(Dataset):
    """Dataset loader for OCT images"""
    def __init__(self, root_dir, split='train', transform=None):
        # Adapter pour la structure: data/oct2017/OCT2017_/train
        base = Path(root_dir)
        
        # Chercher le bon dossier OCT2017_
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
        
        print(f"Using dataset path: {self.root_dir}")
        
        self.transform = transform
        self.classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']  # Classes réelles du dataset
        self.samples = []
        
        for idx, cls in enumerate(self.classes):
            cls_path = self.root_dir / cls
            if cls_path.exists():
                # Chercher tous les formats d'image
                images = list(cls_path.glob('*.jpeg')) + \
                        list(cls_path.glob('*.jpg')) + \
                        list(cls_path.glob('*.png'))
                
                for img_path in images:
                    self.samples.append((str(img_path), idx))
                
                print(f"  {cls}: {len(images)} images")
            else:
                print(f"  WARNING: {cls} folder not found")
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root_dir}")
        
        print(f"Total loaded: {len(self.samples)} images")
    
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
            # Return next valid image
            return self.__getitem__((idx + 1) % len(self))

def get_transforms(split='train'):
    """Get data transforms - SIMPLIFIED for speed"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def create_tiny_model(num_classes=4):
    """Create lightweight model for testing"""
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def create_subset(dataset, samples_per_class=50):
    """Create balanced subset of dataset"""
    class_indices = {i: [] for i in range(4)}
    
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    
    selected_indices = []
    for cls_idx, indices in class_indices.items():
        sampled = random.sample(indices, min(samples_per_class, len(indices)))
        selected_indices.extend(sampled)
    
    random.shuffle(selected_indices)
    return Subset(dataset, selected_indices)

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
    print("QUICK TEST MODE - 5 MINUTES MAX")
    print("=" * 60)
    
    # Configuration ULTRA RAPIDE
    DATA_DIR = 'data'  # Cherchera automatiquement dans data/oct2017/OCT2017_
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    SAMPLES_PER_CLASS = 50
    VAL_SAMPLES = 25
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cpu')
    
    print(f"\n✓ Using device: {DEVICE}")
    print(f"✓ Images per class: {SAMPLES_PER_CLASS} (train) / {VAL_SAMPLES} (val)")
    print(f"✓ Total epochs: {NUM_EPOCHS}")
    print(f"✓ Image size: 128×128 (reduced for speed)")
    
    # Create full datasets
    print("\n" + "-" * 60)
    print("Loading TRAIN dataset...")
    print("-" * 60)
    try:
        train_dataset_full = OCTDataset(DATA_DIR, split='train', 
                                        transform=get_transforms('train'))
    except Exception as e:
        print(f"ERROR: Cannot load train dataset: {e}")
        print("\nPlease check your dataset structure:")
        print("  data/oct2017/OCT2017_/train/CNV/")
        print("  data/oct2017/OCT2017_/train/DME/")
        print("  data/oct2017/OCT2017_/train/DRUSEN/")
        print("  data/oct2017/OCT2017_/train/NORMAL/")
        return
    
    print("\n" + "-" * 60)
    print("Loading TEST dataset...")
    print("-" * 60)
    try:
        val_dataset_full = OCTDataset(DATA_DIR, split='test', 
                                      transform=get_transforms('val'))
    except Exception as e:
        print(f"ERROR: Cannot load test dataset: {e}")
        return
    
    # Create small subsets
    print("\n" + "-" * 60)
    print("Creating subsets...")
    print("-" * 60)
    train_subset = create_subset(train_dataset_full, SAMPLES_PER_CLASS)
    val_subset = create_subset(val_dataset_full, VAL_SAMPLES)
    
    print(f"✓ Train subset: {len(train_subset)} images")
    print(f"✓ Val subset: {len(val_subset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
    
    # Create model
    print("\n" + "-" * 60)
    print("Creating model...")
    print("-" * 60)
    model = create_tiny_model(num_classes=4)
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING QUICK TRAINING")
    print("=" * 60)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )
        
        print(f"\n📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Save test model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_dataset_full.classes,
        'val_acc': val_acc
    }, 'models/quick_test_model.pth')
    
    print("\n" + "=" * 60)
    print("✅ QUICK TEST COMPLETED!")
    print("=" * 60)
    print(f"✓ Model saved to: models/quick_test_model.pth")
    print(f"✓ Final validation accuracy: {val_acc:.2f}%")
    print(f"\nClasses: {train_dataset_full.classes}")
    print("\nℹ️  This is a PROOF OF CONCEPT with limited data.")
    print("   For production, run train.py with full dataset.")
    print("=" * 60)
    
    # Test prediction
    print("\n" + "=" * 60)
    print("TESTING PREDICTION")
    print("=" * 60)
    
    test_img_path, test_label = val_dataset_full.samples[0]
    print(f"Test image: {Path(test_img_path).name}")
    print(f"True label: {train_dataset_full.classes[test_label]}")
    
    model.eval()
    test_transform = get_transforms('val')
    test_img = Image.open(test_img_path).convert('RGB')
    test_tensor = test_transform(test_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(test_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    
    pred_class = train_dataset_full.classes[pred.item()]
    print(f"Predicted: {pred_class} (confidence: {conf.item()*100:.2f}%)")
    
    print("\n✅ Pipeline works! Ready to train full model with train.py")

if __name__ == '__main__':
    main()