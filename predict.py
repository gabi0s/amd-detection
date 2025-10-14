"""
AMD OCT Classification - Prediction Script
CLI tool for predicting AMD from OCT images
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import sys
from pathlib import Path
import numpy as np

class AMDPredictor:
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint.get('classes', ['CNV', 'DME', 'DRUSEN', 'NORMAL'])
        
        # Create model
        self.model = self._create_model(len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("✓ Model loaded successfully")
    
    def _create_model(self, num_classes):
        """Create model architecture"""
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0()
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        return model.to(self.device)
    
    def predict(self, image_path):
        """Predict diagnosis for a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            pred_class = self.classes[predicted.item()]
            conf_score = confidence.item() * 100
            
            # Get all probabilities
            all_probs = {
                cls: prob.item() * 100 
                for cls, prob in zip(self.classes, probabilities[0])
            }
            
            return {
                'prediction': pred_class,
                'confidence': conf_score,
                'probabilities': all_probs
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_report(self, result, image_path):
        """Generate clinical report"""
        if 'error' in result:
            return f"ERROR: {result['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("AI-GUIDED MACULAR ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nImage: {Path(image_path).name}")
        report.append(f"\nPRIMARY DIAGNOSIS: {result['prediction']}")
        report.append(f"Confidence: {result['confidence']:.2f}%")
        
        # Risk assessment
        report.append("\n" + "-" * 60)
        report.append("RISK ASSESSMENT:")
        report.append("-" * 60)
        
        pred = result['prediction']
        conf = result['confidence']
        
        # Map diagnosis to risk level
        if pred in ['CNV', 'DME']:  # Pathologies sérieuses
            if conf > 90:
                risk = "HIGH RISK"
                action = "URGENT specialist referral recommended"
            elif conf > 70:
                risk = "MODERATE RISK"
                action = "Specialist review recommended within 1-2 weeks"
            else:
                risk = "LOW-MODERATE RISK"
                action = "Consider specialist consultation"
        elif pred == 'DRUSEN':  # Early AMD indicator
            if conf > 80:
                risk = "MODERATE RISK"
                action = "Monitor closely, consider specialist review"
            else:
                risk = "LOW-MODERATE RISK"
                action = "Routine monitoring recommended"
        elif pred == 'NORMAL':
            risk = "LOW RISK"
            action = "Routine monitoring recommended"
        else:
            risk = "REQUIRES EVALUATION"
            action = f"Detected {pred} - specialist review needed"
        
        report.append(f"Risk Level: {risk}")
        report.append(f"Recommended Action: {action}")
        
        # Clinical notes
        report.append("\n" + "-" * 60)
        report.append("CLINICAL NOTES:")
        report.append("-" * 60)
        
        clinical_info = {
            'CNV': 'Choroidal Neovascularization - wet AMD, requires immediate attention',
            'DME': 'Diabetic Macular Edema - requires diabetic retinopathy management',
            'DRUSEN': 'Drusen deposits detected - early indicator of AMD',
            'NORMAL': 'No pathological features detected'
        }
        
        report.append(clinical_info.get(pred, 'Unknown condition'))
        
        # Probability breakdown
        report.append("\n" + "-" * 60)
        report.append("DETAILED PROBABILITY BREAKDOWN:")
        report.append("-" * 60)
        
        sorted_probs = sorted(
            result['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for cls, prob in sorted_probs:
            bar = "█" * int(prob / 5)
            report.append(f"{cls:10s}: {prob:5.2f}% {bar}")
        
        # Disclaimer
        report.append("\n" + "=" * 60)
        report.append("IMPORTANT NOTICE:")
        report.append("This is an AI-assisted preliminary analysis tool.")
        report.append("Final diagnosis must be confirmed by a qualified physician.")
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(
        description='AMD OCT Image Classification CLI'
    )
    parser.add_argument(
        'image',
        help='Path to OCT image file'
    )
    parser.add_argument(
        '--model',
        default='models/best_model.pth',
        help='Path to model checkpoint (default: models/best_model.pth)'
    )
    parser.add_argument(
        '--output',
        help='Save report to file (optional)'
    )
    parser.add_argument(
        '--batch',
        help='Process multiple images from a directory'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        print("Please train the model first using train.py or quick_test.py")
        sys.exit(1)
    
    # Initialize predictor
    predictor = AMDPredictor(args.model)
    
    # Process single image
    if not args.batch:
        if not Path(args.image).exists():
            print(f"ERROR: Image file not found: {args.image}")
            sys.exit(1)
        
        print(f"\nAnalyzing: {args.image}")
        print("Please wait...\n")
        
        result = predictor.predict(args.image)
        report = predictor.generate_report(result, args.image)
        
        print(report)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Report saved to: {args.output}")
    
    # Process batch
    else:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"ERROR: Directory not found: {args.batch}")
            sys.exit(1)
        
        image_files = list(batch_dir.glob('*.jpeg')) + \
                     list(batch_dir.glob('*.jpg')) + \
                     list(batch_dir.glob('*.png'))
        
        print(f"\nFound {len(image_files)} images")
        print("Processing batch...\n")
        
        results = []
        for img_path in image_files:
            result = predictor.predict(str(img_path))
            results.append((img_path.name, result))
            status = result.get('prediction', 'ERROR')
            conf = result.get('confidence', 0)
            print(f"✓ {img_path.name}: {status} ({conf:.1f}%)")
        
        # Summary
        print("\n" + "=" * 60)
        print("BATCH SUMMARY")
        print("=" * 60)
        
        cnv_count = sum(1 for _, r in results if r.get('prediction') == 'CNV')
        dme_count = sum(1 for _, r in results if r.get('prediction') == 'DME')
        drusen_count = sum(1 for _, r in results if r.get('prediction') == 'DRUSEN')
        normal_count = sum(1 for _, r in results if r.get('prediction') == 'NORMAL')
        
        print(f"Total images: {len(results)}")
        print(f"CNV (wet AMD): {cnv_count}")
        print(f"DME: {dme_count}")
        print(f"DRUSEN (early AMD): {drusen_count}")
        print(f"NORMAL: {normal_count}")
        print(f"Errors: {len(results) - cnv_count - dme_count - drusen_count - normal_count}")

if __name__ == '__main__':
    main()