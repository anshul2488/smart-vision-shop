#!/usr/bin/env python3
"""
Handwritten OCR Integration for Flask App
Integrates the trained CRNN model with the existing OCR pipeline
"""

import torch
import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Dict, Optional
import logging

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from models.handwritten_ocr_model import HandwrittenOCRModel, HybridOCRProcessor
    HANDWRITTEN_OCR_AVAILABLE = True
except ImportError as e:
    HANDWRITTEN_OCR_AVAILABLE = False
    print(f"⚠️ Handwritten OCR not available: {e}")

class HandwrittenOCRIntegration:
    """Integration class for handwritten OCR model in Flask app"""
    
    def __init__(self, model_path: str = "models/best_model.pth"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.hybrid_processor = None
        
        # Initialize components
        self._initialize_model()
        self._initialize_hybrid_processor()
    
    def _initialize_model(self):
        """Initialize the trained CRNN model"""
        if not HANDWRITTEN_OCR_AVAILABLE:
            print("⚠️ Handwritten OCR not available, using fallback")
            return
        
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ Model not found at {self.model_path}, using fallback")
                return
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Load vocabulary
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = checkpoint['idx_to_char']
            num_classes = checkpoint['num_classes']
            model_config = checkpoint['model_config']
            
            # Initialize model with correct architecture (always 512, 3 layers)
            self.model = HandwrittenOCRModel(
                num_classes=num_classes,
                img_height=model_config['img_height'],
                img_width=model_config['img_width'],
                hidden_size=512,  # Always use 512 to match saved model
                rnn_layers=3  # Always use 3 layers to match saved model
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Handwritten OCR model loaded from: {self.model_path}")
            print(f"📊 Vocabulary size: {num_classes}")
            print(f"🔧 Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading handwritten OCR model: {e}")
            self.model = None
    
    def _initialize_hybrid_processor(self):
        """Initialize hybrid OCR processor"""
        if not HANDWRITTEN_OCR_AVAILABLE:
            return
        
        try:
            self.hybrid_processor = HybridOCRProcessor(use_gpu=torch.cuda.is_available())
            print("✅ Hybrid OCR processor initialized")
        except Exception as e:
            print(f"⚠️ Hybrid OCR processor initialization failed: {e}")
            self.hybrid_processor = None
    
    def preprocess_image_for_crnn(self, image_path: str) -> torch.Tensor:
        """Preprocess image for CRNN model"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for handwritten text
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply adaptive thresholding for better text contrast
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Resize to model input size (64x512)
        image = cv2.resize(image, (512, 64))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add channel dimension
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def predict_crnn(self, image_path: str) -> str:
        """Predict text using CRNN model only"""
        if not self.model:
            return ""
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image_for_crnn(image_path)
            
            with torch.no_grad():
                # Forward pass
                logits = self.model(image_tensor)
                
                # Greedy decoding
                predicted_indices = torch.argmax(logits, dim=2).squeeze(0)
                
                # Decode to text
                predicted_text = self._decode_text(predicted_indices)
            
            return predicted_text
            
        except Exception as e:
            print(f"CRNN prediction failed: {e}")
            return ""
    
    def predict_hybrid(self, image_path: str) -> Tuple[str, float]:
        """Predict text using hybrid EasyOCR + CRNN approach"""
        if not self.hybrid_processor:
            # Fallback to CRNN only
            crnn_text = self.predict_crnn(image_path)
            return crnn_text, 0.5
        
        try:
            # Get EasyOCR prediction
            easyocr_text, easyocr_confidence = self.hybrid_processor.extract_with_easyocr(image_path)
            
            # Get CRNN prediction
            crnn_text = self.predict_crnn(image_path)
            
            # Combine predictions based on confidence
            if easyocr_confidence > 0.7:
                # High confidence EasyOCR - use it as primary
                if crnn_text and len(crnn_text) > 0:
                    # Use CRNN as validation/correction
                    combined_text = f"{easyocr_text} [CRNN: {crnn_text}]"
                else:
                    combined_text = easyocr_text
                final_confidence = easyocr_confidence
            elif easyocr_confidence > 0.3:
                # Medium confidence EasyOCR - combine with CRNN
                if crnn_text and len(crnn_text) > 0:
                    combined_text = f"{easyocr_text} | {crnn_text}"
                else:
                    combined_text = easyocr_text
                final_confidence = (easyocr_confidence + 0.5) / 2  # Average confidence
            else:
                # Low confidence EasyOCR - rely on CRNN
                combined_text = crnn_text if crnn_text else easyocr_text
                final_confidence = 0.5 if crnn_text else easyocr_confidence
            
            return combined_text, final_confidence
            
        except Exception as e:
            print(f"Hybrid prediction failed: {e}")
            # Fallback to CRNN only
            crnn_text = self.predict_crnn(image_path)
            return crnn_text, 0.5
    
    def _decode_text(self, indices: torch.Tensor) -> str:
        """Decode indices to text using CTC decoding"""
        if not self.idx_to_char:
            return ""
        
        # Remove consecutive duplicates and blank tokens
        decoded_indices = []
        prev_idx = -1
        
        for idx in indices:
            idx = idx.item()
            if idx != prev_idx and idx != self.char_to_idx.get('<BLANK>', 0):
                decoded_indices.append(idx)
            prev_idx = idx
        
        # Convert to text
        text = ''.join([self.idx_to_char[idx] for idx in decoded_indices if idx in self.idx_to_char])
        
        return text
    
    def is_available(self) -> bool:
        """Check if handwritten OCR is available"""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "model_path": self.model_path,
            "device": str(self.device),
            "vocabulary_size": len(self.char_to_idx) if self.char_to_idx else 0,
            "hybrid_processor": self.hybrid_processor is not None
        }

# Global instance
handwritten_ocr = HandwrittenOCRIntegration()
