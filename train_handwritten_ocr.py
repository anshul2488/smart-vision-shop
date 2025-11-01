#!/usr/bin/env python3
"""
Training script for Handwritten OCR Model
Trains a CRNN model on grocery list images with CTC loss

Usage:
    python train_handwritten_ocr.py [--epochs 50] [--batch-size 8] [--dataset-path dataset/train_data]
"""

import os
import sys
import argparse

# Add models directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(
        description='Train Handwritten OCR Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default settings (50 epochs, batch size 8)
    python train_handwritten_ocr.py
    
    # Train for 100 epochs
    python train_handwritten_ocr.py --epochs 100
    
    # Train with larger batch size (requires more GPU memory)
    python train_handwritten_ocr.py --batch-size 16
    
    # Use custom dataset path
    python train_handwritten_ocr.py --dataset-path my_dataset/train
        """
    )
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training (default: 8)')
    parser.add_argument('--dataset-path', type=str, 
                       default=r'C:\Users\Anshul Shinde\Desktop\SEM 7\BTECH\project_pipeline\dataset\train_data',
                       help='Path to dataset directory')
    parser.add_argument('--model-save-path', type=str, default='models',
                       help='Path to save trained models (default: models)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📝 Handwritten OCR Model Training")
    print("=" * 70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model save path: {args.model_save_path}")
    print("=" * 70)
    print()
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print("❌ Error: 'models' directory not found!")
        print(f"   Expected: {models_dir}")
        return
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset directory not found: {args.dataset_path}")
        print("   Please check the path and try again.")
        return
    
    # Import the model
    try:
        from handwritten_ocr_model import HandwrittenOCRTrainer
    except ImportError as e:
        print(f"❌ Error importing handwritten OCR model: {e}")
        print("   Make sure you're in the correct directory and models/handwritten_ocr_model.py exists")
        return
    
    # Initialize trainer
    try:
        print("🚀 Initializing trainer...")
        trainer = HandwrittenOCRTrainer(
            dataset_path=args.dataset_path,
            model_save_path=args.model_save_path,
            batch_size=args.batch_size
        )
        
        # Train the model
        print("\n🎯 Starting training...")
        print("-" * 70)
        trainer.train(epochs=args.epochs)
        print("-" * 70)
        
        # Test on a sample image
        print("\n🧪 Testing on sample image...")
        sample_image = os.path.join(args.dataset_path, "1.png")
        sample_text = os.path.join(args.dataset_path, "1.txt")
        
        if os.path.exists(sample_image) and os.path.exists(sample_text):
            predicted_text = trainer.predict(sample_image, use_hybrid=False)
            with open(sample_text, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip()
            
            print(f"\n📸 Sample Image: {os.path.basename(sample_image)}")
            print(f"📝 Ground Truth: {ground_truth}")
            print(f"🤖 Predicted:    {predicted_text}")
            
            # Calculate similarity
            import difflib
            similarity = difflib.SequenceMatcher(None, ground_truth.lower(), predicted_text.lower()).ratio()
            print(f"🎯 Similarity:   {similarity*100:.1f}%")
        else:
            print("⚠️  Sample image or text file not found, skipping test")
        
        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print(f"📁 Models saved in: {args.model_save_path}")
        print("📊 Training curves saved: models/training_curves.png")
        print(f"🏆 Best model: {args.model_save_path}/best_model.pth")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Partial checkpoint may be available")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
