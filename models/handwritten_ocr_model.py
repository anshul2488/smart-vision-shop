#!/usr/bin/env python3
"""
Handwritten OCR Model for Grocery Lists
Uses CRNN (Convolutional Recurrent Neural Network) with CTC Loss
Optimized for handwritten grocery list recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import json
import re
import string
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# EasyOCR import with PIL compatibility fix
try:
    import easyocr
    # Fix PIL compatibility issue
    import PIL.Image
    if not hasattr(PIL.Image, 'ANTIALIAS'):
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ EasyOCR not available. Install with: pip install easyocr")
except Exception as e:
    EASYOCR_AVAILABLE = False
    print(f"⚠️ EasyOCR initialization failed: {e}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

@dataclass
class OCRSample:
    image_path: str
    ground_truth: str
    image_tensor: torch.Tensor
    text_encoded: torch.Tensor
    text_length: int
    easyocr_text: str = ""
    easyocr_confidence: float = 0.0

class HybridOCRProcessor:
    """Hybrid OCR processor combining EasyOCR + CRNN"""
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.easyocr_reader = None
        
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available() and use_gpu)
                print(f"✅ EasyOCR initialized on {'GPU' if torch.cuda.is_available() and use_gpu else 'CPU'}")
            except Exception as e:
                print(f"⚠️ EasyOCR initialization failed: {e}")
                self.easyocr_reader = None
        else:
            print("⚠️ EasyOCR not available")
    
    def preprocess_for_easyocr(self, image_path: str) -> np.ndarray:
        """Preprocess image specifically for EasyOCR with grayscale optimization"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale for better EasyOCR performance
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhanced preprocessing for handwritten text
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding for better contrast
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_with_easyocr(self, image_path: str) -> Tuple[str, float]:
        """Extract text using EasyOCR with optimized parameters"""
        if not self.easyocr_reader:
            return "", 0.0
        
        try:
            # Preprocess image
            processed_image = self.preprocess_for_easyocr(image_path)
            
            # EasyOCR with optimized parameters for handwritten text
            results = self.easyocr_reader.readtext(
                processed_image,
                width_ths=0.1,  # Lower threshold for better detection
                height_ths=0.1,
                paragraph=False,  # Better for individual text lines
                detail=1
            )
            
            if not results:
                return "", 0.0
            
            # Combine all detected text
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.1:  # Filter low confidence results
                    texts.append(text.strip())
                    confidences.append(confidence)
            
            if not texts:
                return "", 0.0
            
            # Combine text and calculate average confidence
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences)
            
            return combined_text, avg_confidence
            
        except Exception as e:
            # Silent fail for training - don't print every error
            return "", 0.0
    
    def preprocess_for_crnn(self, image_path: str, img_height: int = 64, img_width: int = 512) -> torch.Tensor:
        """Preprocess image for CRNN with grayscale optimization"""
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
        
        # Resize to target dimensions
        image = cv2.resize(image, (img_width, img_height))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add channel dimension
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # Shape: (1, H, W)
        
        return image_tensor

class GroceryOCRDataset(Dataset):
    """Dataset for handwritten grocery OCR training with hybrid EasyOCR + CRNN"""
    
    def __init__(self, dataset_path: str, char_to_idx: Dict[str, int], 
                 img_height: int = 64, img_width: int = 512, 
                 augment: bool = True, use_easyocr: bool = True):
        self.dataset_path = dataset_path
        self.char_to_idx = char_to_idx
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.use_easyocr = use_easyocr
        
        # Initialize hybrid OCR processor
        self.ocr_processor = HybridOCRProcessor(use_gpu=torch.cuda.is_available()) if use_easyocr else None
        
        # Load dataset
        self.samples = self._load_dataset()
        
        # Setup augmentation
        if augment:
            self.augmentation = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.Rotate(limit=5, p=0.5),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
            ])
        else:
            self.augmentation = None
        
        print(f"Loaded {len(self.samples)} samples")
        if use_easyocr and self.ocr_processor:
            print("✅ Hybrid OCR mode: EasyOCR + CRNN")
        else:
            print("📝 CRNN-only mode")
    
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
            # Handle cases like "6 (1)" -> "6"
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
        
        # Get EasyOCR text if available
        easyocr_text = ""
        easyocr_confidence = 0.0
        
        if self.use_easyocr and self.ocr_processor:
            easyocr_text, easyocr_confidence = self.ocr_processor.extract_with_easyocr(image_path)
        
        # Load and preprocess image for CRNN
        image = self._load_and_preprocess_image(image_path)
        
        # Encode text (use ground truth for training)
        text_encoded = self._encode_text(ground_truth)
        
        return OCRSample(
            image_path=image_path,
            ground_truth=ground_truth,
            image_tensor=image,
            text_encoded=text_encoded,
            text_length=len(text_encoded),
            easyocr_text=easyocr_text,
            easyocr_confidence=easyocr_confidence
        )
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for CRNN using hybrid processor"""
        if self.use_easyocr and self.ocr_processor:
            # Use hybrid processor for preprocessing
            return self.ocr_processor.preprocess_for_crnn(
                image_path, self.img_height, self.img_width
            )
        else:
            # Fallback to original preprocessing
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale for better OCR recognition
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing for handwritten text
            image = cv2.GaussianBlur(image, (3, 3), 0)
            image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Resize to target dimensions
            image = cv2.resize(image, (self.img_width, self.img_height))
            
            # Apply augmentation if enabled
            if self.augmentation:
                image = self.augmentation(image=image)['image']
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert to tensor and add channel dimension
            image_tensor = torch.from_numpy(image).unsqueeze(0)  # Shape: (1, H, W)
            
            return image_tensor
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to indices"""
        # Clean text
        text = self._clean_text(text)
        
        # Convert to indices
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Use unknown character index
                indices.append(self.char_to_idx.get('<UNK>', 0))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Keep only printable characters and spaces
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()

class CNNBackbone(nn.Module):
    """CNN backbone for feature extraction - optimized for 64x512 input"""
    
    def __init__(self, input_channels: int = 1, hidden_size: int = 512):
        super().__init__()
        
        # Always output 512 channels to match saved model
        self.conv_layers = nn.Sequential(
            # First conv block - reduce height only
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 32x512
            
            # Second conv block - reduce height only
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 16x512
            
            # Third conv block - reduce height only
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 8x512
            
            # Fourth conv block - reduce height only
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 4x512
            
            # Fifth conv block - reduce height only
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 2x512
            
            # Sixth conv block - reduce height only
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 1x512
            
            # Final conv block - reduce width to get sequence length (always 512 output)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),  # 1x128
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class RNNHead(nn.Module):
    """RNN head for sequence modeling"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size * 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to output classes
        output = self.output_projection(lstm_out)
        
        return output

class HandwrittenOCRModel(nn.Module):
    """Complete CRNN model for handwritten OCR"""
    
    def __init__(self, num_classes: int, img_height: int = 64, img_width: int = 512,
                 hidden_size: int = 512, rnn_layers: int = 3):
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        
        # CNN backbone - always outputs 512 channels to match saved model
        self.cnn_backbone = CNNBackbone(input_channels=1, hidden_size=512)
        
        # RNN head - always use 512 to match saved model
        self.rnn_head = RNNHead(
            input_size=512,  # CNN outputs 512 channels
            hidden_size=512,  # Must match saved model
            num_classes=num_classes,
            num_layers=3  # Always use 3 layers to match saved model
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, height, width)
        batch_size = x.size(0)
        
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)  # (batch_size, hidden_size, 1, width)
        
        # Reshape for RNN: (batch_size, width, hidden_size)
        rnn_input = cnn_features.squeeze(2).transpose(1, 2)
        
        # RNN sequence modeling
        rnn_output = self.rnn_head(rnn_input)  # (batch_size, width, num_classes)
        
        return rnn_output

class CTCLoss(nn.Module):
    """CTC Loss for sequence-to-sequence learning"""
    
    def __init__(self, blank_idx: int = 0):
        super().__init__()
        self.blank_idx = blank_idx
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    
    def forward(self, logits, targets, input_lengths, target_lengths):
        # logits: (batch_size, seq_len, num_classes)
        # targets: (total_target_length,)
        # input_lengths: (batch_size,)
        # target_lengths: (batch_size,)
        
        # Transpose logits for CTC: (seq_len, batch_size, num_classes)
        logits = logits.transpose(0, 1)
        
        # Apply log_softmax
        log_probs = F.log_softmax(logits, dim=2)
        
        # Compute CTC loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        return loss

class HandwrittenOCRTrainer:
    """Trainer for handwritten OCR model with GPU support"""
    
    def __init__(self, dataset_path: str, model_save_path: str = "models", batch_size: int = 8):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        
        # Set device with GPU preference
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"🔧 CUDA Version: {torch.version.cuda}")
            print(f"🔥 PyTorch CUDA: {torch.cuda.is_available()}")
        else:
            self.device = torch.device('cpu')
            print("⚠️ Using CPU (GPU not available)")
            print("💡 To enable GPU training, ensure CUDA is installed and PyTorch supports it")
        
        # Create model directory
        os.makedirs(model_save_path, exist_ok=True)
        
        # Build character vocabulary
        self.char_to_idx, self.idx_to_char = self._build_vocabulary()
        self.num_classes = len(self.char_to_idx)
        
        print(f"Vocabulary size: {self.num_classes}")
        print(f"Characters: {list(self.char_to_idx.keys())[:20]}...")
        
        # Initialize model with architecture matching the saved weights
        self.model = HandwrittenOCRModel(
            num_classes=self.num_classes,
            img_height=64,
            img_width=512,
            hidden_size=512,  # Must match saved model
            rnn_layers=3      # Must match saved model
        ).to(self.device)
        
        # Initialize loss and optimizer with better settings
        self.criterion = CTCLoss(blank_idx=self.char_to_idx['<BLANK>'])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # Initialize datasets with hybrid OCR (disable EasyOCR during training to avoid PIL issues)
        self.train_dataset = GroceryOCRDataset(
            dataset_path, self.char_to_idx, augment=True, use_easyocr=False
        )
        self.val_dataset = GroceryOCRDataset(
            dataset_path, self.char_to_idx, augment=False, use_easyocr=False
        )
        
        # Split dataset
        self._split_dataset()
        
        # Initialize data loaders with optimized settings
        num_workers = 4 if torch.cuda.is_available() else 0
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, 
            collate_fn=self._collate_fn, num_workers=num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=self._collate_fn, num_workers=num_workers, pin_memory=True
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Data loaders initialized with {num_workers} workers")
        
        # Test model architecture
        self._test_model_architecture()
    
    def _test_model_architecture(self):
        """Test if the model architecture works correctly"""
        print("🧪 Testing model architecture...")
        
        try:
            # Create a dummy input
            dummy_input = torch.randn(1, 1, 64, 512).to(self.device)
            
            # Test forward pass
            with torch.no_grad():
                output = self.model(dummy_input)
            
            print(f"✅ Model test successful!")
            print(f"📊 Input shape: {dummy_input.shape}")
            print(f"📊 Output shape: {output.shape}")
            print(f"📊 Expected output: (1, 128, {self.num_classes})")
            
            if output.shape[1] != 128:
                print(f"⚠️ Warning: Expected sequence length 128, got {output.shape[1]}")
            
        except Exception as e:
            print(f"❌ Model architecture test failed: {e}")
            raise e
    
    def _build_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build character vocabulary from dataset"""
        # Load all text files
        text_files = glob.glob(os.path.join(self.dataset_path, "*.txt"))
        
        # Collect all characters
        all_text = ""
        for text_file in text_files:
            with open(text_file, 'r', encoding='utf-8') as f:
                all_text += f.read().lower()
        
        # Get unique characters
        chars = sorted(list(set(all_text)))
        
        # Add special tokens
        special_tokens = ['<BLANK>', '<UNK>', '<START>', '<END>']
        chars = special_tokens + [char for char in chars if char not in special_tokens]
        
        # Create mappings
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        return char_to_idx, idx_to_char
    
    def _split_dataset(self):
        """Split dataset into train and validation"""
        # Use 80% for training, 20% for validation
        total_samples = len(self.train_dataset.samples)
        train_size = int(0.8 * total_samples)
        
        # Shuffle indices
        indices = list(range(total_samples))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # Split
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create new datasets
        self.train_dataset.samples = [self.train_dataset.samples[i] for i in train_indices]
        self.val_dataset.samples = [self.val_dataset.samples[i] for i in val_indices]
    
    def _collate_fn(self, batch):
        """Custom collate function for CTC training"""
        images = torch.stack([sample.image_tensor for sample in batch])
        
        # Concatenate all target sequences
        targets = torch.cat([sample.text_encoded for sample in batch])
        
        # Get input and target lengths
        input_lengths = torch.tensor([128] * len(batch))  # Sequence length after CNN (512/4 = 128)
        target_lengths = torch.tensor([sample.text_length for sample in batch])
        
        return {
            'images': images,
            'targets': targets,
            'input_lengths': input_lengths,
            'target_lengths': target_lengths,
            'samples': batch
        }
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['images'].to(self.device)
            targets = batch['targets'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images)
            
            # Compute loss
            loss = self.criterion(logits, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)
                
                # Forward pass
                logits = self.model(images)
                
                # Compute loss
                loss = self.criterion(logits, targets, input_lengths, target_lengths)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs: int = 50):
        """Train the model"""
        print(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"{self.model_save_path}/best_model.pth")
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Log progress
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f"{self.model_save_path}/checkpoint_epoch_{epoch+1}.pth")
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_losses)
        
        print("Training completed!")
    
    def save_model(self, path: str):
        """Save model and vocabulary"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'num_classes': self.num_classes,
            'model_config': {
                'img_height': 64,
                'img_width': 512,
                'hidden_size': 512,
                'rnn_layers': 3,
                'cnn_output_channels': 512,
                'rnn_input_size': 512,
                'rnn_hidden_size': 512
            }
        }, path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load model and vocabulary"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        
        print(f"Model loaded from: {path}")
    
    def _plot_training_curves(self, train_losses: List[float], val_losses: List[float]):
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.model_save_path}/training_curves.png")
        plt.close()
    
    def predict(self, image_path: str, use_hybrid: bool = True) -> str:
        """Predict text from image using hybrid EasyOCR + CRNN approach"""
        if use_hybrid:
            # Create a new OCR processor for inference
            try:
                ocr_processor = HybridOCRProcessor(use_gpu=torch.cuda.is_available())
                return self._predict_hybrid_with_processor(image_path, ocr_processor)
            except Exception as e:
                print(f"Hybrid prediction failed, falling back to CRNN: {e}")
                return self._predict_crnn_only(image_path)
        else:
            return self._predict_crnn_only(image_path)
    
    def _predict_hybrid_with_processor(self, image_path: str, ocr_processor: HybridOCRProcessor) -> str:
        """Hybrid prediction using EasyOCR + CRNN with external processor"""
        # Get EasyOCR prediction
        easyocr_text, easyocr_confidence = ocr_processor.extract_with_easyocr(image_path)
        
        # Get CRNN prediction
        crnn_text = self._predict_crnn_only(image_path)
        
        # Combine predictions based on confidence
        if easyocr_confidence > 0.7:
            # High confidence EasyOCR - use it as primary
            if crnn_text and len(crnn_text) > 0:
                # Use CRNN as validation/correction
                return f"{easyocr_text} [CRNN: {crnn_text}]"
            else:
                return easyocr_text
        elif easyocr_confidence > 0.3:
            # Medium confidence EasyOCR - combine with CRNN
            if crnn_text and len(crnn_text) > 0:
                return f"{easyocr_text} | {crnn_text}"
            else:
                return easyocr_text
        else:
            # Low confidence EasyOCR - rely on CRNN
            return crnn_text if crnn_text else easyocr_text
    
    def _predict_crnn_only(self, image_path: str) -> str:
        """CRNN-only prediction"""
        self.model.eval()
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image, (512, 64))
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            
            # Decode using greedy decoding
            predicted_indices = torch.argmax(logits, dim=2).squeeze(0)
            
            # Convert to text
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

def main():
    """Main function to train the model"""
    print("Handwritten OCR Model Training")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print()
    
    # Check dataset
    dataset_path = "dataset/train_data"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    try:
        # Initialize trainer
        trainer = HandwrittenOCRTrainer(dataset_path)
        
        # Train the model
        trainer.train(epochs=50)
        
        # Test on a sample image
        sample_image = os.path.join(dataset_path, "1.png")
        if os.path.exists(sample_image):
            predicted_text = trainer.predict(sample_image)
            print(f"\nSample prediction:")
            print(f"Image: {sample_image}")
            print(f"Predicted: {predicted_text}")
            
            # Show ground truth
            with open(os.path.join(dataset_path, "1.txt"), 'r') as f:
                ground_truth = f.read().strip()
            print(f"Ground truth: {ground_truth}")
        
        print(f"\n✅ Training completed!")
        print(f"📁 Models saved in: models/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
