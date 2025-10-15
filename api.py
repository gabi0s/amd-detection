"""
AMD OCT Classification - REST API
API Flask pour l'analyse d'images OCT
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
from pathlib import Path
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin pour l'interface web

# Configuration
MODEL_PATH = 'models/best_model.pth'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class AMDPredictor:
    """Classe pour charger et utiliser le modèle"""
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint.get('classes', ['CNV', 'DME', 'DRUSEN', 'NORMAL'])
        
        # Create model
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0()
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, len(self.classes))
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model.to(self.device)
        
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
    
    def predict(self, image):
        """Predict diagnosis from PIL Image"""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        pred_class = self.classes[predicted.item()]
        conf_score = confidence.item() * 100
        
        all_probs = {
            cls: round(prob.item() * 100, 2)
            for cls, prob in zip(self.classes, probabilities[0])
        }
        
        return {
            'prediction': pred_class,
            'confidence': round(conf_score, 2),
            'probabilities': all_probs,
            'risk_level': self._assess_risk(pred_class, conf_score),
            'clinical_note': self._get_clinical_note(pred_class),
            'recommended_action': self._get_action(pred_class, conf_score)
        }
    
    def _assess_risk(self, prediction, confidence):
        """Évaluer le niveau de risque"""
        if prediction in ['CNV', 'DME']:
            if confidence > 90:
                return 'HIGH'
            elif confidence > 70:
                return 'MODERATE'
            else:
                return 'LOW-MODERATE'
        elif prediction == 'DRUSEN':
            return 'MODERATE' if confidence > 80 else 'LOW-MODERATE'
        elif prediction == 'NORMAL':
            return 'LOW'
        return 'UNKNOWN'
    
    def _get_clinical_note(self, prediction):
        """Obtenir la note clinique"""
        notes = {
            'CNV': 'Choroidal Neovascularization detected - wet AMD, requires immediate attention',
            'DME': 'Diabetic Macular Edema - requires diabetic retinopathy management',
            'DRUSEN': 'Drusen deposits detected - early indicator of AMD progression',
            'NORMAL': 'No pathological features detected in the scan'
        }
        return notes.get(prediction, 'Unknown condition detected')
    
    def _get_action(self, prediction, confidence):
        """Recommandation d'action"""
        if prediction in ['CNV', 'DME']:
            if confidence > 90:
                return 'URGENT specialist referral recommended'
            elif confidence > 70:
                return 'Specialist review recommended within 1-2 weeks'
            else:
                return 'Consider specialist consultation'
        elif prediction == 'DRUSEN':
            return 'Monitor closely, consider specialist review' if confidence > 80 else 'Routine monitoring recommended'
        elif prediction == 'NORMAL':
            return 'Continue routine monitoring'
        return 'Further evaluation required'

# Initialiser le prédicteur au démarrage
try:
    predictor = AMDPredictor(MODEL_PATH)
except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    predictor = None

@app.route('/')
def home():
    """Page d'accueil de l'API"""
    return jsonify({
        'name': 'AMD OCT Analysis API',
        'version': '1.0',
        'status': 'online' if predictor else 'model not loaded',
        'endpoints': {
            '/predict': 'POST - Analyze OCT image',
            '/health': 'GET - Check API health',
            '/model-info': 'GET - Get model information'
        }
    })

@app.route('/health')
def health():
    """Vérifier l'état de l'API"""
    return jsonify({
        'status': 'healthy' if predictor else 'model not loaded',
        'model_loaded': predictor is not None,
        'device': str(predictor.device) if predictor else 'N/A',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info')
def model_info():
    """Informations sur le modèle"""
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'classes': predictor.classes,
        'architecture': 'EfficientNet-B0',
        'input_size': '224x224',
        'device': str(predictor.device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Analyser une image OCT
    
    Accepte:
    - multipart/form-data avec 'image' file
    - application/json avec 'image' en base64
    
    Retourne: Diagnostic JSON avec prédiction et recommandations
    """
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        print("=== PREDICT REQUEST ===")
        print(f"Content-Type: {request.content_type}")
        print(f"Files: {request.files}")
        print(f"Form: {request.form}")
        
        # Récupérer l'image
        image = None
        filename = 'uploaded_image.jpg'
        
        # Option 1: Upload de fichier
        if 'image' in request.files:
            print("Found image in files")
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            filename = file.filename
            print(f"Processing file: {filename}")
            
            # Lire l'image
            image = Image.open(file.stream).convert('RGB')
            print(f"Image loaded: {image.size}")
        
        # Option 2: Base64
        elif request.is_json and 'image' in request.json:
            print("Found image in JSON")
            image_data = request.json['image']
            # Retirer le préfixe data:image/...;base64, si présent
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            print(f"Image loaded from base64: {image.size}")
        
        else:
            print("No image found in request")
            return jsonify({'error': 'No image provided. Send as multipart/form-data with key "image"'}), 400
        
        # Sauvegarder l'image (optionnel)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(UPLOAD_FOLDER) / f"{timestamp}_{filename}"
        image.save(save_path)
        print(f"Image saved to: {save_path}")
        
        # Prédiction
        print("Starting prediction...")
        result = predictor.predict(image)
        print(f"Prediction complete: {result['prediction']}")
        
        # Ajouter des métadonnées
        result['metadata'] = {
            'filename': filename,
            'timestamp': timestamp,
            'image_size': list(image.size),
            'saved_path': str(save_path)
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Analyser plusieurs images en batch
    
    Accepte: multipart/form-data avec plusieurs fichiers
    Retourne: Liste de diagnostics
    """
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    results = []
    
    for file in files:
        try:
            image = Image.open(file.stream).convert('RGB')
            result = predictor.predict(image)
            result['filename'] = file.filename
            results.append({
                'success': True,
                'filename': file.filename,
                'result': result
            })
        except Exception as e:
            results.append({
                'success': False,
                'filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({
        'total': len(files),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'results': results
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("AMD OCT ANALYSIS API")
    print("=" * 60)
    print("\nStarting server...")
    print("API available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /              - API info")
    print("  GET  /health        - Health check")
    print("  GET  /model-info    - Model information")
    print("  POST /predict       - Analyze single image")
    print("  POST /batch-predict - Analyze multiple images")
    print("\n" + "=" * 60)
    
    # Mode debug pour développement, désactiver en production
    app.run(host='0.0.0.0', port=5000, debug=True)