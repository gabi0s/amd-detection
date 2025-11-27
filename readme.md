# AMD OCT Classification System

AI-Powered Age-related Macular Degeneration Detection from Retinal OCT Scans

Deep learning system for automatic classification of OCT retinal scans into 4 diagnostic categories: CNV (wet AMD), DME, DRUSEN (early AMD), and NORMAL. Built with PyTorch and EfficientNet-B0.

Validation Accuracy: 87.3% | Inference: <100ms per image

## Quick Start (Windows)

### 1. Download and Launch (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/amd-oct-classification.git
cd amd-oct-classification

# Install dependencies
pip install -r requirements.txt

# Launch the application
start.bat
```
Access the app at: ` http://localhost:8000 `

# Dataset Setup
## Download the Dataset

- 1.Source: [Kermany et al. OCT2017 Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- 2.Download the dataset (5.2GB)
- 3.Extract the ZIP file

Place in your project directory

```texte
amd-oct-classification/
│
├── data/
│   └── oct2017/
│       └── OCT2017_/             ⬅ Extract dataset here
│           ├── train/              (Training images: ~37,000)
│           │   ├── CNV/            (37,205 images)
│           │   ├── DME/            (11,348 images)
│           │   ├── DRUSEN/         (8,616 images)
│           │   └── NORMAL/         (26,315 images)
│           │
│           └── test/                (Validation images: ~1,000)
│               ├── CNV/            (250 images)
│               ├── DME/            (250 images)
│               ├── DRUSEN/         (250 images)
│               └── NORMAL/         (250 images)
│
├── models/                (Model checkpoints - auto-created)
├── uploads/               (API uploads - auto-created)
├── IHM/                   (Web interface)
└── [Python scripts...]
```
## Verify Dataset Installation

```python
python debug_test.py
```
Expected output:

```text
✓ Dataset found: data/oct2017/OCT2017_/train
  ✓ CNV: 37205 images
  ✓ DME: 11348 images
  ✓ DRUSEN: 8616 images
  ✓ NORMAL: 26315 images
✅ TOUT EST PRÊT!
```

# Model Architecture & Training

## Algorithm: EfficientNet-B0 with Transfer Learning
Why EfficientNet-B0?

-  Lightweight: Only 5.3M parameters (vs 25M for ResNet-50)
-  Efficient: Compound scaling (depth + width + resolution)
-  Accurate: 77.1% ImageNet top-1 accuracy
-  Fast: ~50ms inference on CPU
-  Medical imaging: Proven performance on small datasets

## Architecture:

```text
Input (224×224×3 RGB image)
    ↓
EfficientNet-B0 Backbone (Pretrained on ImageNet)
    ├── MBConv blocks with squeeze-and-excitation
    └── Feature extraction: 1280 dimensions
    ↓
Dropout (20% - prevents overfitting)
    ↓
Fully Connected Layer (1280 → 4 classes)
    ↓
Softmax → [P(CNV), P(DME), P(DRUSEN), P(NORMAL)]
```

## Training Time Estimates

| Mode   | Mode     | Size          | Epochs | Hardware | Time       | Accuracy |
|--------|----------|---------------|--------|----------|------------|----------|
| Quick  | Test     | 200 images    | 3      | CPU      | 5 minutes  | ~75%     |
| Full   | Training | 37,000 images | 20     | CPU      | 8-10 hours | ~87%     |
| Full   | Training | 37,000 images | 20     |GPU (CUDA)| 2-3 hours  |  ~87%  |

# Complete Project Structure

```text
amd-oct-classification/
│
├── DATA
│ └── OCT2017_/ # OCT2017 Dataset
│ ├── test/ # Test images (~1,000)
│ │ ├── CNV/ (250 images)
│ │ ├── DME/ (250 images)
│ │ ├── DRUSEN/ (250 images)
│ │ └── NORMAL/ (250 images)
│ │
│ ├── train/ # Training images (~37,000)
│ │ ├── CNV/ (37,205 images)
│ │ ├── DME/ (11,348 images)
│ │ ├── DRUSEN/ (8,616 images)
│ │ └── NORMAL/ (26,315 images)
│ │
│ └── val/ # Validation images
│ ├── CNV/
│ ├── DME/
│ ├── DRUSEN/
│ └── NORMAL/
│
├── WEB INTERFACE
│ └── IHM/
│ ├── index.html # Main web page
│ ├── style.css # Styling and layout
│ └── app.js # Frontend JavaScript logic
│
├── MODELS
│ ├── best_model.pth # Production model (87% accuracy)
│ ├── quick_test_model.pth # Quick test model (75% accuracy)
│ └── history_20251015_121741.json # Training history and metrics
│
├── UPLOADS
│ └── [user-uploaded-images] # Temporary storage for API uploads
│
├── CORE SCRIPTS
│ ├── api.py # Flask REST API server
│ ├── predict.py # Command-line prediction tool
│ ├── train.py # Full model training script
│ ├── quick_test.py # Quick validation pipeline
│ ├── debug_test.py # Environment diagnostics
│ └── start.bat # Windows auto-launcher
│
├── CONFIGURATION
│ ├── requirements.txt # Python dependencies
│ ├── readme.md # Project documentation
│ └── .gitignore # Git ignore rules
```

# Usage Example
### CLI

```python
# Single image
python predict.py path/to/scan.jpeg

# Batch processing
python predict.py dummy.jpeg --batch path/to/folder/

# Save report to file
python predict.py scan.jpeg --output report.txt
```

### Output Example:

```text
============================================================
AI-GUIDED MACULAR ANALYSIS REPORT
============================================================

Image: patient_scan.jpeg

PRIMARY DIAGNOSIS: CNV
Confidence: 94.32%

RISK ASSESSMENT: HIGH RISK
Recommended Action: URGENT specialist referral recommended

DETAILED PROBABILITY BREAKDOWN:
CNV       : 94.32% ██████████████████
DME       :  3.21% ▌
DRUSEN    :  1.87% ▌
NORMAL    :  0.60% 
============================================================
```

### REST API

```python
# Start API server
python api.py

# Test with cURL
curl -X POST http://localhost:5000/predict -F "image=@scan.jpeg"
```

#### API RESPOND
```json
{
  "success": true,
  "result": {
    "prediction": "CNV",
    "confidence": 94.32,
    "risk_level": "HIGH",
    "recommended_action": "URGENT specialist referral recommended"
  }
}
```

# Model Performance

| Metric           | Value                   |
|------------------|-------------------------|
| Overall Accuracy | 87.3%                   |
| Inference Speed  | 50ms (CPU) / 10ms (GPU) |
| Model Size       | 21 MB                   |


# Web overview

Start by  launching `start.bat`, can take 1 minute to connect the local API

![alt text](/png/image.png)