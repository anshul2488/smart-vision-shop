#!/usr/bin/env python3
"""
Training script for Handwritten OCR Model
Simplified interface for training the CRNN model
"""

import os
import sys
import argparse
import torch
from handwritten_ocr_model import HandwrittenOCRTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Handwritten OCR Model')
    parser.add_argument('--dataset_path', type=str, default='dataset/train_data',
                       help='Path to training dataset')
    parser.add_argument('--model_save_path', type=str, default='models',
                       help='Path to save trained models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print("Handwritten OCR Model Training")
    print("=" * 40)
    print(f"Dataset: {args.dataset_path}")
    print(f"Model save path: {args.model_save_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset not found at: {args.dataset_path}")
        print("Please make sure the dataset directory exists and contains image-text pairs.")
        return
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print()
    
    try:
        # Initialize trainer
        print("Initializing trainer...")
        trainer = HandwrittenOCRTrainer(
            dataset_path=args.dataset_path,
            model_save_path=args.model_save_path,
            batch_size=args.batch_size
        )
        
        # Train the model
        print("Starting training...")
        trainer.train(epochs=args.epochs)
        
        # Test on a sample image
        print("\nTesting on sample image...")
        sample_images = ['1.png', '10.png', '50.png']
        
        for sample_img in sample_images:
            sample_path = os.path.join(args.dataset_path, sample_img)
            if os.path.exists(sample_path):
                print(f"\nTesting: {sample_img}")
                
                # Get prediction
                predicted_text = trainer.predict(sample_path)
                print(f"Predicted: {predicted_text}")
                
                # Show ground truth
                txt_path = os.path.join(args.dataset_path, sample_img.replace('.png', '.txt'))
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        ground_truth = f.read().strip()
                    print(f"Ground truth: {ground_truth}")
                
                # Calculate accuracy (simple character-level)
                if ground_truth:
                    from difflib import SequenceMatcher
                    accuracy = SequenceMatcher(None, predicted_text.lower(), ground_truth.lower()).ratio()
                    print(f"Accuracy: {accuracy:.3f}")
                break
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Models saved in: {args.model_save_path}/")
        print(f"📊 Training curves saved as: {args.model_save_path}/training_curves.png")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
