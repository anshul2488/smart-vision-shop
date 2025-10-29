#!/usr/bin/env python3
"""
Evaluation script for Handwritten OCR Model
Test model performance on the dataset and generate detailed metrics
"""

import os
import sys
import argparse
import torch
import json
import pandas as pd
from difflib import SequenceMatcher
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from handwritten_ocr_model import HandwrittenOCRModel
from inference_handwritten_ocr import HandwrittenOCRInference

class HandwrittenOCREvaluator:
    """Evaluator for handwritten OCR model"""
    
    def __init__(self, model_path: str, dataset_path: str):
        self.model_path = model_path
        self.dataset_path = dataset_path
        
        # Load model
        self.ocr = HandwrittenOCRInference(model_path)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"📊 Dataset loaded: {len(self.dataset)} samples")
    
    def _load_dataset(self):
        """Load dataset with ground truth"""
        dataset = []
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg']
        for ext in image_extensions:
            for filename in os.listdir(self.dataset_path):
                if filename.lower().endswith(ext):
                    image_path = os.path.join(self.dataset_path, filename)
                    
                    # Find corresponding text file
                    base_name = os.path.splitext(filename)[0]
                    base_name = base_name.replace(' (1)', '')  # Handle duplicate names
                    text_path = os.path.join(self.dataset_path, f"{base_name}.txt")
                    
                    if os.path.exists(text_path):
                        with open(text_path, 'r', encoding='utf-8') as f:
                            ground_truth = f.read().strip()
                        
                        dataset.append({
                            'image_path': image_path,
                            'text_path': text_path,
                            'ground_truth': ground_truth,
                            'filename': filename
                        })
        
        return dataset
    
    def evaluate_single(self, sample: dict) -> dict:
        """Evaluate single sample"""
        try:
            # Get prediction
            predicted_text = self.ocr.predict(sample['image_path'])
            
            # Calculate metrics
            ground_truth = sample['ground_truth'].lower().strip()
            predicted = predicted_text.lower().strip()
            
            # Character-level accuracy
            char_accuracy = SequenceMatcher(None, ground_truth, predicted).ratio()
            
            # Word-level accuracy
            gt_words = set(ground_truth.split())
            pred_words = set(predicted.split())
            
            if gt_words:
                word_precision = len(gt_words.intersection(pred_words)) / len(pred_words) if pred_words else 0
                word_recall = len(gt_words.intersection(pred_words)) / len(gt_words)
                word_f1 = 2 * word_precision * word_recall / (word_precision + word_recall) if (word_precision + word_recall) > 0 else 0
            else:
                word_precision = word_recall = word_f1 = 0
            
            # Exact match
            exact_match = 1 if ground_truth == predicted else 0
            
            return {
                'filename': sample['filename'],
                'ground_truth': ground_truth,
                'predicted': predicted,
                'char_accuracy': char_accuracy,
                'word_precision': word_precision,
                'word_recall': word_recall,
                'word_f1': word_f1,
                'exact_match': exact_match,
                'success': True
            }
            
        except Exception as e:
            return {
                'filename': sample['filename'],
                'ground_truth': sample['ground_truth'],
                'predicted': '',
                'char_accuracy': 0,
                'word_precision': 0,
                'word_recall': 0,
                'word_f1': 0,
                'exact_match': 0,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_all(self) -> pd.DataFrame:
        """Evaluate all samples in dataset"""
        print("Running evaluation on all samples...")
        
        results = []
        for i, sample in enumerate(self.dataset):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(self.dataset)}...")
            
            result = self.evaluate_single(sample)
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate overall metrics"""
        successful = df[df['success'] == True]
        
        if len(successful) == 0:
            return {'error': 'No successful predictions'}
        
        metrics = {
            'total_samples': len(df),
            'successful_predictions': len(successful),
            'success_rate': len(successful) / len(df),
            
            # Character-level metrics
            'char_accuracy_mean': successful['char_accuracy'].mean(),
            'char_accuracy_std': successful['char_accuracy'].std(),
            'char_accuracy_min': successful['char_accuracy'].min(),
            'char_accuracy_max': successful['char_accuracy'].max(),
            
            # Word-level metrics
            'word_precision_mean': successful['word_precision'].mean(),
            'word_recall_mean': successful['word_recall'].mean(),
            'word_f1_mean': successful['word_f1'].mean(),
            
            # Exact match
            'exact_match_rate': successful['exact_match'].mean(),
        }
        
        return metrics
    
    def analyze_errors(self, df: pd.DataFrame) -> dict:
        """Analyze common errors"""
        successful = df[df['success'] == True]
        
        # Find samples with low accuracy
        low_accuracy = successful[successful['char_accuracy'] < 0.5]
        
        # Common error patterns
        error_patterns = defaultdict(int)
        
        for _, row in low_accuracy.iterrows():
            gt = row['ground_truth']
            pred = row['predicted']
            
            # Length differences
            if len(gt) != len(pred):
                error_patterns['length_mismatch'] += 1
            
            # Character substitutions
            if len(gt) == len(pred):
                substitutions = sum(1 for a, b in zip(gt, pred) if a != b)
                error_patterns[f'substitutions_{substitutions}'] += 1
        
        return {
            'low_accuracy_samples': len(low_accuracy),
            'error_patterns': dict(error_patterns),
            'worst_samples': low_accuracy.nsmallest(5, 'char_accuracy')[['filename', 'char_accuracy', 'ground_truth', 'predicted']].to_dict('records')
        }
    
    def plot_results(self, df: pd.DataFrame, output_dir: str = 'evaluation_results'):
        """Plot evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        successful = df[df['success'] == True]
        
        # Character accuracy distribution
        plt.figure(figsize=(10, 6))
        plt.hist(successful['char_accuracy'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Character Accuracy')
        plt.ylabel('Frequency')
        plt.title('Distribution of Character Accuracy')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'char_accuracy_distribution.png'))
        plt.close()
        
        # Word-level metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(successful['word_precision'], bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Word Precision')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Word Precision Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(successful['word_recall'], bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Word Recall')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Word Recall Distribution')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(successful['word_f1'], bins=20, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Word F1 Score')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Word F1 Score Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'word_metrics_distribution.png'))
        plt.close()
        
        # Accuracy vs sample index
        plt.figure(figsize=(12, 6))
        plt.plot(successful['char_accuracy'], alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Character Accuracy')
        plt.title('Character Accuracy Across Samples')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'accuracy_across_samples.png'))
        plt.close()
    
    def save_results(self, df: pd.DataFrame, metrics: dict, error_analysis: dict, 
                    output_dir: str = 'evaluation_results'):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save error analysis
        with open(os.path.join(output_dir, 'error_analysis.json'), 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        # Save summary report
        with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
            f.write("Handwritten OCR Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total samples: {metrics['total_samples']}\n")
            f.write(f"Successful predictions: {metrics['successful_predictions']}\n")
            f.write(f"Success rate: {metrics['success_rate']:.3f}\n\n")
            
            f.write("Character-level metrics:\n")
            f.write(f"  Mean accuracy: {metrics['char_accuracy_mean']:.3f}\n")
            f.write(f"  Std deviation: {metrics['char_accuracy_std']:.3f}\n")
            f.write(f"  Min accuracy: {metrics['char_accuracy_min']:.3f}\n")
            f.write(f"  Max accuracy: {metrics['char_accuracy_max']:.3f}\n\n")
            
            f.write("Word-level metrics:\n")
            f.write(f"  Mean precision: {metrics['word_precision_mean']:.3f}\n")
            f.write(f"  Mean recall: {metrics['word_recall_mean']:.3f}\n")
            f.write(f"  Mean F1 score: {metrics['word_f1_mean']:.3f}\n\n")
            
            f.write(f"Exact match rate: {metrics['exact_match_rate']:.3f}\n\n")
            
            f.write("Error Analysis:\n")
            f.write(f"  Low accuracy samples: {error_analysis['low_accuracy_samples']}\n")
            f.write(f"  Error patterns: {error_analysis['error_patterns']}\n")
        
        print(f"✅ Results saved to: {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Handwritten OCR Model')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, default='dataset/train_data',
                       help='Path to evaluation dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("Handwritten OCR Model Evaluation")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at: {args.model_path}")
        print("Please train the model first using train_handwritten_ocr.py")
        return 1
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset not found at: {args.dataset_path}")
        return 1
    
    try:
        # Initialize evaluator
        evaluator = HandwrittenOCREvaluator(args.model_path, args.dataset_path)
        
        # Run evaluation
        print("Running evaluation...")
        df = evaluator.evaluate_all()
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = evaluator.calculate_metrics(df)
        
        # Analyze errors
        print("Analyzing errors...")
        error_analysis = evaluator.analyze_errors(df)
        
        # Plot results
        print("Generating plots...")
        evaluator.plot_results(df, args.output_dir)
        
        # Save results
        print("Saving results...")
        evaluator.save_results(df, metrics, error_analysis, args.output_dir)
        
        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Successful predictions: {metrics['successful_predictions']}")
        print(f"Success rate: {metrics['success_rate']:.3f}")
        print(f"Mean character accuracy: {metrics['char_accuracy_mean']:.3f}")
        print(f"Mean word F1 score: {metrics['word_f1_mean']:.3f}")
        print(f"Exact match rate: {metrics['exact_match_rate']:.3f}")
        print(f"\nResults saved to: {args.output_dir}/")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
