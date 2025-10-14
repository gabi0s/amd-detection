"""
Debug script - Vérifie l'environnement et les dépendances
"""

print("=" * 60)
print("DIAGNOSTIC DE L'ENVIRONNEMENT")
print("=" * 60)

# 1. Version Python
import sys
print(f"\n✓ Python version: {sys.version}")
print(f"✓ Python executable: {sys.executable}")

# 2. Vérifier les imports
print("\n" + "-" * 60)
print("Vérification des dépendances...")
print("-" * 60)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  - CUDA disponible: {torch.cuda.is_available()}")
    print(f"  - Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except ImportError as e:
    print(f"✗ PyTorch: NON INSTALLÉ")
    print(f"  Erreur: {e}")
    print("\n  → Installer avec: pip install torch torchvision")

try:
    import torchvision
    print(f"✓ TorchVision: {torchvision.__version__}")
except ImportError:
    print(f"✗ TorchVision: NON INSTALLÉ")
    print("  → Installer avec: pip install torchvision")

try:
    from PIL import Image
    print(f"✓ Pillow (PIL): OK")
except ImportError:
    print(f"✗ Pillow: NON INSTALLÉ")
    print("  → Installer avec: pip install Pillow")

try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except ImportError:
    print(f"✗ NumPy: NON INSTALLÉ")
    print("  → Installer avec: pip install numpy")

try:
    from tqdm import tqdm
    print(f"✓ tqdm: OK")
except ImportError:
    print(f"✗ tqdm: NON INSTALLÉ")
    print("  → Installer avec: pip install tqdm")

# 3. Vérifier la structure des dossiers
print("\n" + "-" * 60)
print("Vérification de la structure du projet...")
print("-" * 60)

from pathlib import Path

base_dir = Path('.')
data_dir = Path('data/OCT2017')

print(f"\n✓ Dossier actuel: {base_dir.resolve()}")

# Vérifier data/OCT2017
if data_dir.exists():
    print(f"✓ Dataset trouvé: {data_dir.resolve()}")
    
    # Vérifier train/test
    for split in ['train', 'test']:
        split_path = data_dir / split
        if split_path.exists():
            print(f"  ✓ {split}/")
            
            # Vérifier les classes
            for cls in ['NORMAL', 'AMD', 'DME', 'DRUSEN']:
                cls_path = split_path / cls
                if cls_path.exists():
                    count = len(list(cls_path.glob('*.jpeg')))
                    if count > 0:
                        print(f"    ✓ {cls}: {count} images")
                    else:
                        print(f"    ✗ {cls}: VIDE (0 images)")
                else:
                    print(f"    ✗ {cls}: DOSSIER MANQUANT")
        else:
            print(f"  ✗ {split}/: DOSSIER MANQUANT")
else:
    print(f"✗ Dataset NON TROUVÉ: {data_dir.resolve()}")
    print("\n  → Téléchargez le dataset depuis Kaggle:")
    print("     https://www.kaggle.com/datasets/paultimothymooney/kermany2018")
    print("  → Extrayez-le dans: data/OCT2017/")

# 4. Vérifier models/
models_dir = Path('models')
if models_dir.exists():
    print(f"\n✓ Dossier models/: {models_dir.resolve()}")
else:
    print(f"\n✗ Dossier models/: NON TROUVÉ (sera créé automatiquement)")

# 5. Test rapide de création de tenseur
print("\n" + "-" * 60)
print("Test rapide de PyTorch...")
print("-" * 60)

try:
    import torch
    x = torch.rand(1, 3, 128, 128)
    print(f"✓ Création de tenseur: OK")
    print(f"  Shape: {x.shape}")
    print(f"  Device: {x.device}")
except Exception as e:
    print(f"✗ Erreur PyTorch: {e}")

# 6. Résumé
print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)

issues = []

try:
    import torch
except:
    issues.append("PyTorch non installé")

try:
    import torchvision
except:
    issues.append("TorchVision non installé")

try:
    from PIL import Image
except:
    issues.append("Pillow non installé")

if not data_dir.exists():
    issues.append("Dataset manquant")

if issues:
    print("\n⚠️  PROBLÈMES DÉTECTÉS:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n📋 ACTIONS REQUISES:")
    print("\n1. Installer les dépendances:")
    print("   pip install torch torchvision Pillow numpy tqdm")
    print("\n2. Télécharger le dataset:")
    print("   https://www.kaggle.com/datasets/paultimothymooney/kermany2018")
    print("   Extraire dans: data/OCT2017/")
else:
    print("\n✅ TOUT EST PRÊT!")
    print("\nVous pouvez maintenant lancer:")
    print("  python quick_test.py")
    print("  ou")
    print("  python train.py")

print("=" * 60)