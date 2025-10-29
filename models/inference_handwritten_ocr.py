#!/usr/bin/env python3
"""
Inference script for Handwritten OCR Model
Load trained model and predict text from images
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from handwritten_ocr_model import HandwrittenOCRModel, HandwrittenOCRTrainer

class HandwrittenOCRInference:
    """Inference class for trained handwritten OCR model"""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load vocabulary
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.num_classes = checkpoint['num_classes']
        model_config = checkpoint['model_config']
        
        # Initialize model
        self.model = HandwrittenOCRModel(
            num_classes=self.num_classes,
            img_height=model_config['img_height'],
            img_width=model_config['img_width'],
            hidden_size=model_config['hidden_size'],
            rnn_layers=model_config['rnn_layers']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"📊 Vocabulary size: {self.num_classes}")
        print(f"🔧 Device: {self.device}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        image = cv2.resize(image, (512, 64))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def predict(self, image_path: str) -> str:
        """Predict text from image"""
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            # Forward pass
            logits = self.model(image_tensor)
            
            # Greedy decoding
            predicted_indices = torch.argmax(logits, dim=2).squeeze(0)
            
            # Decode to text
            predicted_text = self._decode_text(predicted_indices)
        
        return predicted_text
    
    def _decode_text(self, indices: torch.Tensor) -> str:
        """Decode indices to text using CTC decoding"""
        # Remove consecutive duplicates and blank tokens
        decoded_indices = []
        prev_idx = -1
        
        for idx in indices:
            idx = idx.item()
            if idx != prev_idx and idx != self.char_to_idx['<BLANK>']:
                decoded_indices.append(idx)
            prev_idx = idx
        
        # Convert to text
        text = ''.join([self.idx_to_char[idx] for idx in decoded_indices])
        
        return text
    
    def predict_batch(self, image_paths: list) -> list:
        """Predict text from multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                predicted_text = self.predict(image_path)
                results.append({
                    'image_path': image_path,
                    'predicted_text': predicted_text,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'predicted_text': '',
                    'success': False,
                    'error': str(e)
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Handwritten OCR Inference')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--output_file', type=str, default='predictions.txt',
                       help='Output file for predictions')
    
    args = parser.parse_args()
    
    print("Handwritten OCR Inference")
    print("=" * 30)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at: {args.model_path}")
        print("Please train the model first using train_handwritten_ocr.py")
        return 1
    
    try:
        # Load model
        print("Loading model...")
        ocr = HandwrittenOCRInference(args.model_path)
        
        if args.image_path:
            # Single image prediction
            if not os.path.exists(args.image_path):
                print(f"❌ Image not found: {args.image_path}")
                return 1
            
            print(f"Predicting: {args.image_path}")
            predicted_text = ocr.predict(args.image_path)
            
            print(f"Predicted text:")
            print(predicted_text)
            
            # Save to file
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(f"Image: {args.image_path}\n")
                f.write(f"Predicted: {predicted_text}\n")
            
            print(f"✅ Prediction saved to: {args.output_file}")
        
        elif args.image_dir:
            # Batch prediction
            if not os.path.exists(args.image_dir):
                print(f"❌ Directory not found: {args.image_dir}")
                return 1
            
            # Find all images
            image_extensions = ['.png', '.jpg', '.jpeg']
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend([
                    os.path.join(args.image_dir, f) 
                    for f in os.listdir(args.image_dir) 
                    if f.lower().endswith(ext)
                ])
            
            if not image_paths:
                print(f"❌ No images found in: {args.image_dir}")
                return 1
            
            print(f"Found {len(image_paths)} images")
            print("Running batch prediction...")
            
            # Predict
            results = ocr.predict_batch(image_paths)
            
            # Save results
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Image: {result['image_path']}\n")
                    if result['success']:
                        f.write(f"Predicted: {result['predicted_text']}\n")
                    else:
                        f.write(f"Error: {result['error']}\n")
                    f.write("-" * 50 + "\n")
            
            # Print summary
            successful = sum(1 for r in results if r['success'])
            print(f"✅ Batch prediction completed!")
            print(f"📊 Successful: {successful}/{len(results)}")
            print(f"📁 Results saved to: {args.output_file}")
        
        else:
            print("❌ Please provide either --image_path or --image_dir")
            return 1
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
