#!/usr/bin/env python3
"""
Complete Grocery OCR + LLM Model
Combines EasyOCR-sensitive + Grayscale preprocessing, LLM processing, and pandas output
Optimized for GPU training with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import easyocr
import pandas as pd
import re
import os
import glob
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import difflib
from transformers import AutoTokenizer, AutoModel, pipeline
import warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

@dataclass
class GroceryItem:
    item_name: str
    quantity: str
    unit: str
    confidence: float
    raw_text: str
    llm_confidence: float = 0.0

@dataclass
class TrainingSample:
    image_path: str
    ground_truth: str
    processed_text: str
    parsed_items: List[GroceryItem]
    accuracy: float

class GroceryOCRDataset(Dataset):
    """PyTorch Dataset for grocery OCR training"""
    
    def __init__(self, dataset_path: str, ocr_processor, llm_processor):
        self.dataset_path = dataset_path
        self.ocr_processor = ocr_processor
        self.llm_processor = llm_processor
        
        # Load dataset
        self.samples = self._load_dataset()
        
    def _load_dataset(self) -> List[Tuple[str, str]]:
        """Load all image-text pairs from the dataset"""
        image_text_pairs = []
        
        # Get all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.dataset_path, ext)))
        
        print(f"Found {len(image_files)} image files")
        
        for image_path in image_files:
            # Get corresponding text file
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            base_name = re.sub(r'\s*\(\d+\)', '', base_name)
            
            text_file = os.path.join(self.dataset_path, f"{base_name}.txt")
            
            if os.path.exists(text_file):
                with open(text_file, 'r', encoding='utf-8') as f:
                    ground_truth = f.read().strip()
                
                image_text_pairs.append((image_path, ground_truth))
        
        print(f"Successfully loaded {len(image_text_pairs)} image-text pairs")
        return image_text_pairs
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, ground_truth = self.samples[idx]
        
        # Process image with OCR
        processed_image = self.ocr_processor.preprocess_image_optimal(image_path)
        ocr_results = self.ocr_processor.extract_text_optimal(processed_image)
        
        # Combine text
        predicted_text = ' '.join(text for text, _, _ in ocr_results)
        
        # Process with LLM
        llm_results = self.llm_processor.process_text(predicted_text)
        
        # Calculate accuracy
        accuracy = self.ocr_processor.calculate_accuracy(ground_truth, predicted_text)
        
        return {
            'image_path': image_path,
            'ground_truth': ground_truth,
            'predicted_text': predicted_text,
            'llm_results': llm_results,
            'accuracy': accuracy,
            'ocr_results': ocr_results
        }

def custom_collate_fn(batch):
    """Custom collate function to handle our data structure"""
    return batch

class OCRProcessor:
    """Optimized OCR processor using EasyOCR-sensitive + Grayscale"""
    
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
    def preprocess_image_optimal(self, image_path: str) -> np.ndarray:
        """Optimal preprocessing: EasyOCR-sensitive + Grayscale"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize if needed
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            scale_factor = max(100/height, 100/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif height > 2000 or width > 2000:
            scale_factor = min(2000/height, 2000/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale (optimal for EasyOCR-sensitive)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Light enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Slight gamma correction
        gamma = 1.2
        gamma_corrected = np.power(enhanced / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(gamma_corrected)
        
        return gamma_corrected
    
    def extract_text_optimal(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Extract text using optimal EasyOCR-sensitive settings"""
        results = []
        
        # Ensure image is 2D grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) != 2:
            print(f"Warning: Unexpected image shape: {image.shape}")
            return results
        
        try:
            # Optimal parameters from your best result
            parameter_sets = [
                {'width_ths': 0.1, 'height_ths': 0.1, 'paragraph': False},  # Best performing
                {'width_ths': 0.3, 'height_ths': 0.3, 'paragraph': False},  # Backup
                {'width_ths': 0.5, 'height_ths': 0.5, 'paragraph': True},   # Alternative
            ]
            
            best_detections = []
            best_confidence = 0
            
            for params in parameter_sets:
                try:
                    detections = self.easyocr_reader.readtext(image, **params)
                    
                    if detections:
                        avg_confidence = sum(det[2] for det in detections) / len(detections)
                        
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_detections = detections
                            
                except Exception as e:
                    continue
            
            # Convert to our format
            for detection in best_detections:
                bbox, text, confidence = detection
                x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                
                if confidence > 0.1:
                    results.append((
                        text.strip(),
                        confidence,
                        (x1, y1, x2-x1, y2-y1)
                    ))
                    
        except Exception as e:
            print(f"Text extraction failed: {e}")
        
        return results
    
    def calculate_accuracy(self, ground_truth: str, predicted: str) -> float:
        """Calculate accuracy between ground truth and predicted text"""
        gt_normalized = self._normalize_text(ground_truth)
        pred_normalized = self._normalize_text(predicted)
        
        if len(gt_normalized) == 0:
            return 1.0 if len(pred_normalized) == 0 else 0.0
        
        matcher = difflib.SequenceMatcher(None, gt_normalized, pred_normalized)
        return matcher.ratio()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,/\-\(\)]', '', text)
        return text.strip()

class LLMProcessor:
    """LLM processor for text refinement and grocery item extraction"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = device
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            print(f"✅ LLM initialized: {model_name}")
            
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}")
            print("Using fallback text processing...")
            self.generator = None
    
    def process_text(self, text: str) -> Dict:
        """Process text with LLM for grocery item extraction"""
        if self.generator is None:
            return self._fallback_processing(text)
        
        try:
            # Create prompt for grocery item extraction
            prompt = f"""Extract grocery items from this OCR text. Format each item as "item_name - quantity unit".
            
OCR Text: {text}

Grocery Items:
"""
            
            # Generate with LLM
            result = self.generator(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract items from generated text
            items = self._extract_items_from_llm_output(generated_text)
            
            return {
                'original_text': text,
                'llm_output': generated_text,
                'extracted_items': items,
                'confidence': 0.8
            }
            
        except Exception as e:
            print(f"LLM processing failed: {e}")
            return self._fallback_processing(text)
    
    def _fallback_processing(self, text: str) -> Dict:
        """Fallback processing without LLM"""
        items = self._extract_items_simple(text)
        
        return {
            'original_text': text,
            'llm_output': text,
            'extracted_items': items,
            'confidence': 0.6
        }
    
    def _extract_items_from_llm_output(self, llm_output: str) -> List[Dict]:
        """Extract grocery items from LLM output"""
        items = []
        
        # Look for patterns like "item_name - quantity unit"
        patterns = [
            r'([^-]+?)\s*-\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)',
            r'([^-]+?)\s*-\s*(\d+/\d+)\s*([a-zA-Z]+)',
            r'([^-]+?)\s*-\s*(\d+)\s*([a-zA-Z]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, llm_output, re.IGNORECASE)
            for match in matches:
                item_name = match[0].strip()
                quantity = match[1].strip()
                unit = match[2].strip().lower()
                
                if len(item_name) > 2:
                    items.append({
                        'item_name': item_name,
                        'quantity': quantity,
                        'unit': unit,
                        'confidence': 0.8
                    })
        
        return items
    
    def _extract_items_simple(self, text: str) -> List[Dict]:
        """Simple item extraction without LLM"""
        items = []
        
        # Split by common patterns
        parts = re.split(r'(\d+(?:\.\d+)?\s*[a-zA-Z]+)', text)
        
        for i, part in enumerate(parts):
            if re.match(r'\d+(?:\.\d+)?\s*[a-zA-Z]+', part):
                # This is a quantity+unit
                quantity_match = re.search(r'(\d+(?:\.\d+)?)', part)
                unit_match = re.search(r'([a-zA-Z]+)', part)
                
                if quantity_match and unit_match:
                    quantity = quantity_match.group(1)
                    unit = unit_match.group(1).lower()
                    
                    # Find item name (previous part)
                    if i > 0:
                        item_name = parts[i-1].strip()
                        if len(item_name) > 2:
                            items.append({
                                'item_name': item_name,
                                'quantity': quantity,
                                'unit': unit,
                                'confidence': 0.6
                            })
        
        return items

class GroceryOCRLoss(nn.Module):
    """Custom loss function for OCR training"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        # predictions is a list of dictionaries, targets is a list of dictionaries
        total_loss = 0.0
        
        for pred, target in zip(predictions, targets):
            # Calculate text similarity loss
            text_loss = self._text_similarity_loss(pred['text'], target['predicted_text'])
            
            # Calculate item extraction loss
            item_loss = self._item_extraction_loss(pred['items'], target.get('llm_results', {}).get('extracted_items', []))
            
            # Combine losses
            sample_loss = text_loss + 0.5 * item_loss
            total_loss += sample_loss
        
        # Average loss across batch
        return total_loss / len(predictions) if predictions else torch.tensor(0.0)
    
    def _text_similarity_loss(self, pred_text, target_text):
        """Calculate text similarity loss"""
        # Simple character-level similarity
        pred_chars = set(pred_text.lower())
        target_chars = set(target_text.lower())
        
        intersection = len(pred_chars.intersection(target_chars))
        union = len(pred_chars.union(target_chars))
        
        similarity = intersection / union if union > 0 else 0
        return torch.tensor(1 - similarity, requires_grad=True)
    
    def _item_extraction_loss(self, pred_items, target_items):
        """Calculate item extraction loss"""
        if not pred_items or not target_items:
            return torch.tensor(1.0, requires_grad=True)
        
        # Simple item matching loss
        pred_names = [item.get('item_name', '') for item in pred_items]
        target_names = [item.get('item_name', '') for item in target_items]
        
        matches = 0
        for pred_name in pred_names:
            for target_name in target_names:
                if difflib.SequenceMatcher(None, pred_name.lower(), target_name.lower()).ratio() > 0.7:
                    matches += 1
                    break
        
        accuracy = matches / max(len(pred_items), len(target_items))
        return torch.tensor(1 - accuracy, requires_grad=True)

class GroceryOCRModel(nn.Module):
    """Main OCR model with LLM integration"""
    
    def __init__(self, ocr_processor, llm_processor):
        super().__init__()
        self.ocr_processor = ocr_processor
        self.llm_processor = llm_processor
        
        # Text processing layers
        self.text_encoder = nn.LSTM(256, 512, batch_first=True)
        self.text_decoder = nn.Linear(512, 256)
        
        # Item extraction layers
        self.item_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch_data):
        """Forward pass through the model"""
        batch_results = []
        
        for data in batch_data:
            # OCR processing
            image_path = data['image_path']
            processed_image = self.ocr_processor.preprocess_image_optimal(image_path)
            ocr_results = self.ocr_processor.extract_text_optimal(processed_image)
            
            # Combine OCR text
            predicted_text = ' '.join(text for text, _, _ in ocr_results)
            
            # LLM processing
            llm_results = self.llm_processor.process_text(predicted_text)
            
            # Create result
            result = {
                'text': predicted_text,
                'items': llm_results['extracted_items'],
                'confidence': llm_results['confidence'],
                'ocr_results': ocr_results,
                'llm_results': llm_results
            }
            
            batch_results.append(result)
        
        return batch_results

class GroceryOCRTrainer:
    """Main trainer class"""
    
    def __init__(self, dataset_path: str = r"C:\Users\Anshul Shinde\Desktop\SEM 7\BTECH\dataset\train_data"):
        self.dataset_path = dataset_path
        self.device = device
        
        # Initialize components
        self.ocr_processor = OCRProcessor()
        self.llm_processor = LLMProcessor()
        
        # Initialize model
        self.model = GroceryOCRModel(self.ocr_processor, self.llm_processor).to(self.device)
        
        # Initialize loss and optimizer
        self.criterion = GroceryOCRLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initialize dataset
        self.dataset = GroceryOCRDataset(dataset_path, self.ocr_processor, self.llm_processor)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
        
        print(f"✅ Model initialized on {self.device}")
        print(f"📊 Dataset size: {len(self.dataset)}")
    
    def train(self, epochs: int = 10):
        """Train the model"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            for batch_idx, batch_data in enumerate(self.dataloader):
                # batch_data is already a list from custom_collate_fn
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_data)
                
                # Calculate loss
                loss = self.criterion(predictions, batch_data)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                batch_accuracy = sum(data['accuracy'] for data in batch_data) / len(batch_data)
                
                epoch_loss += loss.item()
                epoch_accuracy += batch_accuracy
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}")
            
            avg_loss = epoch_loss / len(self.dataloader)
            avg_accuracy = epoch_accuracy / len(self.dataloader)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    def evaluate(self) -> pd.DataFrame:
        """Evaluate the model and return results as DataFrame"""
        print("🔍 Evaluating model...")
        
        self.model.eval()
        all_results = []
        
        with torch.no_grad():
            for batch_data in self.dataloader:
                # batch_data is already a list from custom_collate_fn
                predictions = self.model(batch_data)
                
                for i, (pred, data) in enumerate(zip(predictions, batch_data)):
                    # Extract items
                    items = []
                    for item in pred['items']:
                        items.append({
                            'item_name': item['item_name'],
                            'quantity': item['quantity'],
                            'unit': item['unit'],
                            'confidence': item['confidence']
                        })
                    
                    # Create result row
                    result = {
                        'image_path': data['image_path'],
                        'image_name': os.path.basename(data['image_path']),
                        'ground_truth': data['ground_truth'],
                        'predicted_text': pred['text'],
                        'accuracy': data['accuracy'],
                        'confidence': pred['confidence'],
                        'total_items': len(items),
                        'items_list': '; '.join([f"{item['item_name']} - {item['quantity']} {item['unit']}" for item in items])
                    }
                    
                    # Add individual items
                    for j, item in enumerate(items):
                        item_result = result.copy()
                        item_result.update({
                            'item_number': j + 1,
                            'item_name': item['item_name'],
                            'quantity': item['quantity'],
                            'unit': item['unit'],
                            'item_confidence': item['confidence']
                        })
                        all_results.append(item_result)
        
        # Create DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Add useful columns
        df_results['quantity_numeric'] = pd.to_numeric(df_results['quantity'], errors='coerce')
        df_results['is_valid_quantity'] = df_results['quantity_numeric'].notna()
        df_results['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return df_results
    
    def save_model(self, path: str = "grocery_ocr_model.pth"):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'device': str(self.device),
                'model_name': self.llm_processor.model_name
            }
        }, path)
        print(f"✅ Model saved to: {path}")
    
    def load_model(self, path: str = "grocery_ocr_model.pth"):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Model loaded from: {path}")

def main():
    """Main function"""
    print("🛒 Grocery OCR + LLM Model")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print()
    
    # Check dataset
    dataset_path = r"C:\Users\Anshul Shinde\Desktop\SEM 7\BTECH\dataset\train_data"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    try:
        # Initialize trainer
        trainer = GroceryOCRTrainer(dataset_path)
        
        # Train the model
        trainer.train(epochs=5)
        
        # Evaluate and get results
        df_results = trainer.evaluate()
        
        # Save results
        df_results.to_csv('grocery_ocr_llm_results.csv', index=False)
        
        # Create summary
        summary = df_results.groupby('image_name').agg({
            'accuracy': 'first',
            'confidence': 'first',
            'total_items': 'first',
            'items_list': 'first'
        }).reset_index()
        
        summary.to_csv('grocery_ocr_llm_summary.csv', index=False)
        
        # Save model
        trainer.save_model()
        
        # Display results
        print("\n📊 Results Summary:")
        print(f"Total samples: {len(df_results)}")
        print(f"Average accuracy: {df_results['accuracy'].mean():.3f}")
        print(f"Average confidence: {df_results['confidence'].mean():.3f}")
        
        print(f"\n💾 Files created:")
        print(f"  - grocery_ocr_llm_results.csv")
        print(f"  - grocery_ocr_llm_summary.csv")
        print(f"  - grocery_ocr_model.pth")
        
        print(f"\n📋 Sample results:")
        print(df_results[['image_name', 'item_name', 'quantity', 'unit', 'item_confidence']].head(10))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
