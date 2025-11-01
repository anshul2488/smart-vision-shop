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

# Try to import pyspellchecker, fallback to simple matching if not available
try:
    from spellchecker import SpellChecker
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False

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
    """Optimized OCR processor using EasyOCR-sensitive + Grayscale with advanced CV techniques"""
    
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        # Common grocery items for intelligent splitting
        self.common_items = {
            'milk', 'ghee', 'sugar', 'basmati', 'rice', 'wheat', 'flour', 'atta',
            'onion', 'onions', 'potato', 'potatoes', 'tomato', 'tomatoes',
            'turmeric', 'chili', 'chilli', 'red chili', 'chili powder',
            'salt', 'oil', 'butter', 'bread', 'eggs', 'curd', 'paneer',
            'cucumber', 'mushroom', 'mushrooms', 'lettuce', 'bell pepper', 'peppers',
            'tofu', 'honey', 'oats', 'pasta', 'herbs', 'oregano', 'basil',
            'detergent', 'cleaner', 'soap', 'shampoo', 'toothpaste',
            'biscuit', 'biscuits', 'cookies', 'coffee', 'tea', 'juice'
        }
        
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
        """Extract text using optimal EasyOCR with advanced CV techniques"""
        results = []
        
        # Ensure image is 2D grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) != 2:
            print(f"Warning: Unexpected image shape: {image.shape}")
            return results
        
        try:
            # Technique 1: Layout Analysis - Detect item boxes/cards using contour detection
            item_groups = self._detect_layout_boxes(image)
            
            # Technique 2: Extract text with spatial grouping
            if item_groups:
                print(f"📦 Detected {len(item_groups)} layout boxes using CV")
                results = self._extract_with_spatial_grouping(image, item_groups)
            else:
                # Fallback to standard extraction
                results = self._extract_standard_ocr(image)
            
            # Technique 3: Post-process with semantic grouping
            results = self._semantic_text_grouping(results)
                    
        except Exception as e:
            print(f"Text extraction failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            results = self._extract_standard_ocr(image)
        
        return results
    
    def _detect_layout_boxes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Technique 1: Layout Analysis - Detect item boxes/cards using contour detection"""
        try:
            # Binarize image
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours (potential boxes/cards)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            height, width = image.shape
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size - look for rectangular regions that could be item cards
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # Valid box criteria: reasonable size and aspect ratio
                if (area > width * height * 0.02 and  # At least 2% of image
                    area < width * height * 0.8 and  # Not too large
                    aspect_ratio > 0.3 and aspect_ratio < 10 and  # Reasonable aspect ratio
                    w > 50 and h > 20):  # Minimum dimensions
                    
                    boxes.append((x, y, w, h))
            
            # Sort by vertical position (top to bottom)
            boxes.sort(key=lambda b: (b[1], b[0]))
            
            return boxes
        except Exception as e:
            print(f"Layout detection failed: {e}")
            return []
    
    def _extract_with_spatial_grouping(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Technique 2: Extract text with spatial grouping - process each box separately"""
        all_results = []
        
        try:
            # Extract text from each detected box
            for box in boxes:
                x, y, w, h = box
                
                # Extract region of interest
                roi = image[max(0, y-10):min(image.shape[0], y+h+10),
                           max(0, x-10):min(image.shape[1], x+w+10)]
                
                if roi.size == 0:
                    continue
                
                # Extract text from this region
                try:
                    detections = self.easyocr_reader.readtext(
                        roi,
                        width_ths=0.3,
                        height_ths=0.3,
                        paragraph=False
                    )
                    
                    for detection in detections:
                        bbox, text, confidence = detection
                        # Adjust coordinates to full image
                        roi_x1, roi_y1 = int(bbox[0][0]), int(bbox[0][1])
                        roi_x2, roi_y2 = int(bbox[2][0]), int(bbox[2][1])
                        
                        global_x1 = x + roi_x1 - 10
                        global_y1 = y + roi_y1 - 10
                        global_w = roi_x2 - roi_x1
                        global_h = roi_y2 - roi_y1
                        
                        if confidence > 0.1:
                            all_results.append((
                                text.strip(),
                                confidence,
                                (max(0, global_x1), max(0, global_y1), max(1, global_w), max(1, global_h))
                            ))
                except Exception as e:
                    continue
            
            # If box-based extraction didn't work well, fallback to full image
            if not all_results or len(all_results) < 2:
                return self._extract_standard_ocr(image)
            
            return all_results
            
        except Exception as e:
            print(f"Spatial grouping extraction failed: {e}")
            return self._extract_standard_ocr(image)
    
    def _extract_standard_ocr(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Standard OCR extraction"""
        results = []
        
        try:
            # Optimal parameters
            parameter_sets = [
                {'width_ths': 0.1, 'height_ths': 0.1, 'paragraph': False},
                {'width_ths': 0.3, 'height_ths': 0.3, 'paragraph': False},
                {'width_ths': 0.5, 'height_ths': 0.5, 'paragraph': True},
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
            print(f"Standard OCR extraction failed: {e}")
        
        return results
    
    def _semantic_text_grouping(self, ocr_results: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Technique 3: Semantic grouping - group text by proximity and similarity"""
        if not ocr_results:
            return ocr_results
        
        # Group by vertical position (y-coordinate) - items on same line
        grouped = {}
        tolerance = 30  # pixels
        
        for text, conf, (x, y, w, h) in ocr_results:
            # Find group with similar y-coordinate
            found_group = False
            for group_y in grouped.keys():
                if abs(y - group_y) <= tolerance:
                    grouped[group_y].append((text, conf, (x, y, w, h)))
                    found_group = True
                    break
            
            if not found_group:
                grouped[y] = [(text, conf, (x, y, w, h))]
        
        # Sort groups by y-position and within groups by x-position
        final_results = []
        for y_pos in sorted(grouped.keys()):
            group_items = grouped[y_pos]
            group_items.sort(key=lambda item: item[2][0])  # Sort by x
            final_results.extend(group_items)
        
        return final_results
    
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

class TextCorrector:
    """Text correction for OCR errors - like Google's spell checker"""
    
    def __init__(self):
        # Common grocery item corrections (OCR errors -> correct)
        self.item_corrections = {
            # Common misspellings
            'basmai': 'basmati',
            'basmti': 'basmati',
            'basmat': 'basmati',
            'grigina': 'original',
            'orginal': 'original',
            'orginl': 'original',
            'orignal': 'original',
            'lpack': 'pack',
            'pak': 'pack',
            'pck': 'pack',
            'pousdec': 'powder',
            'pousedec': 'powder',
            'powdr': 'powder',
            'powde': 'powder',
            'chilli': 'chili',
            'chili': 'chili',
            'chilies': 'chili',
            'redchilli': 'red chili',
            'redchili': 'red chili',
            'tamatar': 'tomato',
            'tamato': 'tomato',
            'tamatr': 'tomato',
            'pyaz': 'onion',
            'pyaaz': 'onion',
            'aloo': 'potato',
            'alo': 'potato',
            'dahi': 'curd',
            'doodh': 'milk',
            'dudh': 'milk',
            'makhan': 'butter',
            'chawal': 'rice',
            'gehun': 'wheat',
            'atta': 'flour',
            'chini': 'sugar',
            'namak': 'salt',
            'nimbu': 'lemon',
            'adrak': 'ginger',
            'lehsun': 'garlic',
            'paneer': 'cheese',
            'chai': 'tea',
            'kapi': 'coffee',
            'namkeen': 'chips',
            'sabun': 'soap',
            'shampo': 'shampoo',
            # Common product name patterns
            'bisciut': 'biscuit',
            'biscuts': 'biscuit',
            'biscut': 'biscuit',
            'biscuits': 'biscuit',
            'biscket': 'biscuit',
            'pacle': 'parle',  # Common brand name typo
            'parleg': 'parle-g',
        }
        
        # Unit corrections (common OCR errors)
        self.unit_corrections = {
            'sun': 'kg',
            'kehirooc': 'kg',
            'kgrooc': 'kg',
            'kgirooc': 'kg',
            'kilrooc': 'kg',
            'kilgrooc': 'kg',
            'q': 'kg',  # Common ambiguous unit, default to kg
            'k': 'kg',
            'kg': 'kg',
            'kilogram': 'kg',
            'kilo': 'kg',
            'g': 'g',
            'gram': 'g',
            'gm': 'g',
            'grams': 'g',
            'l': 'l',
            'liter': 'l',
            'litre': 'l',
            'ml': 'ml',
            'milliliter': 'ml',
            'millilitre': 'ml',
            'pack': 'pack',
            'packs': 'pack',
            'pkt': 'pack',
            'packet': 'pack',
            'pcs': 'pieces',
            'pc': 'pieces',
            'piece': 'pieces',
            'pieces': 'pieces',
            'dozen': 'dozen',
            'dz': 'dozen',
            'box': 'box',
            'boxes': 'box',
            'bottle': 'bottle',
            'bottles': 'bottle',
            'tin': 'tin',
            'tins': 'tin',
            'can': 'can',
            'cans': 'can',
            'bag': 'bag',
            'bags': 'bag',
            'loaf': 'loaf',
            'loaves': 'loaf',
        }
        
        # Initialize spell checker if available
        self.spell_checker = None
        if SPELLCHECK_AVAILABLE:
            try:
                self.spell_checker = SpellChecker()
                # Add grocery terms to dictionary
                grocery_terms = list(self.item_corrections.values())
                for term in grocery_terms:
                    self.spell_checker.word_frequency.load_words([term])
            except Exception as e:
                print(f"Warning: Spell checker initialization failed: {e}")
                self.spell_checker = None
    
    def correct_item_name(self, item_name: str) -> str:
        """Correct grocery item name with spell checking"""
        if not item_name or len(item_name.strip()) < 2:
            return item_name
        
        original = item_name.strip()
        # Normalize: replace underscores and special chars with spaces
        normalized = re.sub(r'[_\-\W]+', ' ', original)
        corrected = normalized.lower().strip()
        
        # First, check direct corrections (exact match)
        if corrected in self.item_corrections:
            return self.item_corrections[corrected].title()
        
        # Check for partial matches in multi-word items
        words = corrected.split()
        corrected_words = []
        has_correction = False
        
        for word in words:
            original_word = word
            # Clean word (remove special chars but preserve for reconstruction)
            word_clean = re.sub(r'[^\w]', '', word).lower()
            
            # Skip very short words, numbers, and single chars
            if len(word_clean) <= 2 or word_clean.isdigit():
                corrected_words.append(original_word)
                continue
            
            word_corrected = None
            
            # Check direct correction
            if word_clean in self.item_corrections:
                word_corrected = self.item_corrections[word_clean]
                has_correction = True
            # Check fuzzy match with grocery items first (higher priority)
            else:
                best_match = self._fuzzy_match_word(word_clean, list(self.item_corrections.keys()))
                if best_match and difflib.SequenceMatcher(None, word_clean, best_match).ratio() > 0.7:
                    word_corrected = self.item_corrections[best_match]
                    has_correction = True
                # Then try spell checker if available
                elif self.spell_checker:
                    try:
                        correction = self.spell_checker.correction(word_clean)
                        if correction and correction != word_clean:
                            # Prefer known grocery terms
                            if correction in self.item_corrections.values():
                                word_corrected = correction
                                has_correction = True
                            # Otherwise use spell checker correction
                            else:
                                word_corrected = correction
                                has_correction = True
                    except Exception:
                        pass
            
            # Use corrected word if found, otherwise keep original
            if word_corrected:
                corrected_words.append(word_corrected)
            else:
                corrected_words.append(original_word)
        
        # Reconstruct the corrected name
        corrected_name = ' '.join(corrected_words)
        
        # Only apply capitalization if we made corrections
        if has_correction:
            corrected_name = ' '.join(word.capitalize() for word in corrected_name.split())
        
        # If correction is too different, return original (preserve original case)
        similarity = difflib.SequenceMatcher(None, original.lower(), corrected_name.lower()).ratio()
        if similarity < 0.5 and not has_correction:
            return original
        
        return corrected_name
    
    def correct_unit(self, unit: str) -> str:
        """Correct unit name"""
        if not unit:
            return unit
        
        unit_lower = unit.lower().strip()
        
        # Direct match
        if unit_lower in self.unit_corrections:
            return self.unit_corrections[unit_lower]
        
        # Fuzzy match for similar units
        best_match = self._fuzzy_match_word(unit_lower, list(self.unit_corrections.keys()))
        if best_match:
            similarity = difflib.SequenceMatcher(None, unit_lower, best_match).ratio()
            if similarity > 0.6:
                return self.unit_corrections[best_match]
        
        # Default to original if no good match
        return unit
    
    def _fuzzy_match_word(self, word: str, candidates: List[str]) -> Optional[str]:
        """Find best fuzzy match from candidates"""
        if not word or not candidates:
            return None
        
        best_match = None
        best_ratio = 0.0
        
        for candidate in candidates:
            ratio = difflib.SequenceMatcher(None, word, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        
        return best_match if best_ratio > 0.6 else None
    
    def correct_text(self, text: str) -> str:
        """Correct entire text (for raw OCR output)"""
        if not text:
            return text
        
        # Split into words and correct
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Remove punctuation for checking
            word_clean = re.sub(r'[^\w]', '', word)
            
            if word_clean:
                corrected_word = self.correct_item_name(word_clean)
                # Preserve original punctuation
                if word != word_clean:
                    # Add punctuation back
                    for char in word:
                        if not char.isalnum():
                            corrected_word += char
                
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)

class LLMProcessor:
    """LLM processor for text refinement and grocery item extraction"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = device
        
        # Initialize text corrector for spell checking
        self.text_corrector = TextCorrector()
        
        # Common grocery items for intelligent splitting
        self.common_items = {
            'milk', 'ghee', 'sugar', 'basmati', 'rice', 'wheat', 'flour', 'atta',
            'onion', 'onions', 'potato', 'potatoes', 'tomato', 'tomatoes',
            'turmeric', 'chili', 'chilli', 'red', 'powder', 'pepper', 'peppers',
            'salt', 'oil', 'butter', 'bread', 'eggs', 'curd', 'paneer', 'tofu',
            'cucumber', 'mushroom', 'mushrooms', 'lettuce', 'bell',
            'honey', 'oats', 'pasta', 'herbs', 'oregano', 'basil',
            'detergent', 'cleaner', 'soap', 'shampoo', 'toothpaste',
            'biscuit', 'biscuits', 'cookies', 'coffee', 'tea', 'juice',
            'cheese', 'canned', 'brown', 'peanut', 'butter',
            'surf', 'excel', 'harpic', 'dettol', 'scotch', 'brite', 'colin', 'goodknight'
        }
        
        # Skip LLM initialization - we use pure OCR extraction only
        # No need to load tokenizer/model/generator since we don't use LLM
        self.generator = None
        self.tokenizer = None
        self.model = None
        
        print("✅ LLMProcessor initialized (LLM disabled, using pure OCR extraction)")
    
    def _intelligent_item_splitting(self, text: str) -> List[str]:
        """Intelligently split concatenated items using CV and semantic analysis"""
        items = []
        
        # Step 1: Pre-process text - clean OCR artifacts
        cleaned = re.sub(r'[~=_|]+', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Step 2: Detect common grocery item boundaries
        words = cleaned.split()
        
        # Use class-level common items
        grocery_keywords = self.common_items
        
        # Step 3: Advanced splitting using multiple strategies
        
        # Strategy A: Split by known item boundaries
        # Look for patterns where known items appear consecutively
        items_found = []
        word_list = cleaned.split()
        
        i = 0
        current_item_words = []
        
        while i < len(word_list):
            word_lower = word_list[i].lower().strip()
            next_word_lower = word_list[i+1].lower().strip() if i+1 < len(word_list) else ''
            
            # Check if current word is a known grocery item
            is_item_word = any(keyword in word_lower or word_lower in keyword for keyword in grocery_keywords)
            is_unit = word_lower in ['kg', 'g', 'gm', 'grams', 'l', 'litre', 'liter', 'ml', 'pack', 'packs', 'bottle', 'can', 'jar', 'box', 'loaf', 'piece', 'pieces', 'head', 'each']
            is_quantity = word_lower.isdigit() or re.match(r'^\d+\.\d+$', word_lower)
            is_number_prefix = re.match(r'^[lux]?(\d+)', word_lower)  # Patterns like "x5", "U5", "L5"
            
            # If we found a complete item pattern: [item words] [quantity] [unit]
            if is_unit and current_item_words:
                # Check if we have a quantity before the unit
                has_quantity = False
                for j in range(max(0, len(current_item_words)-3), len(current_item_words)):
                    if re.match(r'^\d', current_item_words[j]):
                        has_quantity = True
                        break
                
                if has_quantity or len(current_item_words) >= 2:
                    # Complete item found
                    item_text = ' '.join(current_item_words + [word_list[i]])
                    items_found.append(item_text)
                    current_item_words = []
                    i += 1
                    continue
            
            # If we find a known item keyword and we have accumulated words
            if is_item_word and current_item_words:
                # Check if accumulated words form a valid item (has quantity/unit or is long enough)
                has_qty_unit = any(w.isdigit() or w.lower() in ['kg', 'g', 'pack', 'bottle', 'can'] for w in current_item_words)
                
                if has_qty_unit or len(current_item_words) >= 3:
                    # Previous item is complete, save it
                    items_found.append(' '.join(current_item_words))
                    current_item_words = [word_list[i]]
                else:
                    # Continue building current item
                    current_item_words.append(word_list[i])
            elif is_item_word:
                # Start new item
                current_item_words = [word_list[i]]
            elif is_quantity or is_number_prefix:
                # Number found - add to current item
                current_item_words.append(word_list[i])
            else:
                # Regular word
                if current_item_words:
                    current_item_words.append(word_list[i])
                else:
                    # Might be start of an item name
                    current_item_words = [word_list[i]]
            
            i += 1
        
        # Add remaining item
        if current_item_words:
            items_found.append(' '.join(current_item_words))
        
        potential_items = items_found
        
        # Step 4: Filter and validate items
        for item in potential_items:
            item_clean = item.strip()
            if len(item_clean) > 2:
                # Check if it contains at least one known keyword
                item_lower = item_clean.lower()
                if any(keyword in item_lower for keyword in grocery_keywords):
                    items.append(item_clean)
        
        # Final fallback: if no items found, return at least one item with the cleaned text
        if not items:
            # Try one more time with aggressive splitting
            segments = re.split(r'[~=_]+\s*[Uu]?\s*', cleaned_text)
            for segment in segments:
                segment = segment.strip()
                if len(segment) > 3:
                    extracted = self._extract_from_text_segment(segment, seen_items)
                    items.extend(extracted)
            
            # If still nothing, return the cleaned text as one item
            if not items and cleaned.strip():
                items.append({
                    'item_name': self.text_corrector.correct_item_name(cleaned.strip()),
                    'quantity': '1',
                    'unit': 'pack',
                    'confidence': 0.5,
                    'original_item_name': cleaned.strip(),
                    'original_unit': 'pack'
                })
        
        return items
    
    def _extract_from_text_segment(self, text: str, seen_items: set) -> List[Dict]:
        """Extract items from a text segment"""
        items = []
        
        # Clean the segment
        text = re.sub(r'[~=_|]+', ' ', text).strip()
        text = ' '.join(text.split())
        
        if len(text) < 3:
            return items
        
        # Try to find quantity and unit
        # Pattern: item_name [quantity] [unit]
        patterns = [
            r'^(.+?)\s+(\d+(?:\.\d+)?)\s+([a-zA-Z]+)$',  # "Item 5 kg"
            r'^(.+?)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z]+)',  # "Item 5kg"
            r'^(.+?)\s+([a-zA-Z]+)\s+(\d+)',  # "Item kg 5" (reversed)
        ]
        
        item_name = text
        quantity = '1'
        unit = 'pack'
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    # Standard pattern
                    if re.match(r'^\d', match.group(2)):  # Group 2 is quantity
                        item_name = match.group(1).strip()
                        quantity = match.group(2).strip()
                        unit = match.group(3).strip().lower()
                    else:  # Reversed pattern
                        item_name = match.group(1).strip()
                        unit = match.group(2).strip().lower()
                        quantity = match.group(3).strip()
                break
        
        # Clean item name
        item_name = re.sub(r'^[~=_LUI]+\s*|\s*[~=_LUI]+$', '', item_name).strip()
        item_name = ' '.join(item_name.split())
        
        # Clean unit - remove L/U prefix
        unit = re.sub(r'^[lu]', '', unit.lower())
        
        if len(item_name) > 2:
            item_key = item_name.lower()
            if item_key not in seen_items:
                seen_items.add(item_key)
                
                corrected_item_name = self.text_corrector.correct_item_name(item_name)
                corrected_unit = self.text_corrector.correct_unit(unit)
                
                items.append({
                    'item_name': corrected_item_name,
                    'quantity': quantity,
                    'unit': corrected_unit,
                    'confidence': 0.7,
                    'original_item_name': item_name,
                    'original_unit': unit
                })
        
        return items
    
    def process_text(self, text: str) -> Dict:
        """Process text for grocery item extraction - PURE OCR EXTRACTION, NO LLM"""
        print(f"🔍 Pure OCR extraction (NO LLM): {text[:150]}...")
        
        # PURE OCR EXTRACTION - NO LLM AT ALL
        items = self._extract_items_simple(text)
        
        print(f"✅ OCR extraction found {len(items)} items")
        
        return {
            'original_text': text,
            'llm_output': 'OCR extraction only (LLM disabled)',
            'extracted_items': items,
            'confidence': 0.75 if items else 0.5
        }
    
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
        """Extract grocery items from LLM output - multiple format support"""
        items = []
        
        # First, try to extract from structured formats (with dash)
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
                    # Apply text correction to item name and unit
                    corrected_item_name = self.text_corrector.correct_item_name(item_name)
                    corrected_unit = self.text_corrector.correct_unit(unit)
                    
                    items.append({
                        'item_name': corrected_item_name,
                        'quantity': quantity,
                        'unit': corrected_unit,
                        'confidence': 0.8,
                        'original_item_name': item_name,
                        'original_unit': unit
                    })
        
        # If no items found with dash format, try more flexible patterns
        if not items:
            # Pattern: "item_name quantity unit" or "item_name x quantity unit"
            flexible_patterns = [
                r'([a-zA-Z][a-zA-Z\s]+?)\s+(?:x\s+)?(\d+(?:\.\d+)?)\s+([a-zA-Z]+)',
                r'([a-zA-Z][a-zA-Z\s]+?)\s+(\d+(?:\.\d+)?)\s+([a-zA-Z]+)',
                r'([a-zA-Z][a-zA-Z]+)\s+(\d+(?:\.\d+)?)\s+([a-zA-Z]+)',
            ]
            
            for pattern in flexible_patterns:
                matches = re.findall(pattern, llm_output, re.IGNORECASE)
                for match in matches:
                    item_name = match[0].strip()
                    quantity = match[1].strip()
                    unit = match[2].strip().lower()
                    
                    # Skip if too short or looks like a prompt
                    if len(item_name) > 2 and 'grocery' not in item_name.lower() and 'item' not in item_name.lower():
                        corrected_item_name = self.text_corrector.correct_item_name(item_name)
                        corrected_unit = self.text_corrector.correct_unit(unit)
                        
                        items.append({
                            'item_name': corrected_item_name,
                            'quantity': quantity,
                            'unit': corrected_unit,
                            'confidence': 0.7,
                            'original_item_name': item_name,
                            'original_unit': unit
                        })
        
        # If still no items, try line-by-line parsing (common in OCR output)
        if not items:
            lines = llm_output.split('\n')
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Skip prompt-like lines
                if any(word in line.lower() for word in ['grocery', 'items', 'extract', 'ocr text', 'item_name']):
                    continue
                
                # Try to find quantity and unit in the line
                qty_unit_match = re.search(r'(\d+(?:\.\d+)?)\s+([a-zA-Z]+)', line)
                if qty_unit_match:
                    quantity = qty_unit_match.group(1)
                    unit = qty_unit_match.group(2).lower()
                    # Get item name (everything before the quantity)
                    item_name = line[:qty_unit_match.start()].strip()
                    
                    if len(item_name) > 2:
                        corrected_item_name = self.text_corrector.correct_item_name(item_name)
                        corrected_unit = self.text_corrector.correct_unit(unit)
                        
                        items.append({
                            'item_name': corrected_item_name,
                            'quantity': quantity,
                            'unit': corrected_unit,
                            'confidence': 0.65,
                            'original_item_name': item_name,
                            'original_unit': unit
                    })
        
        return items
    
    def _extract_items_simple(self, text: str) -> List[Dict]:
        """Simple item extraction - DIRECT parsing of numbered lists"""
        items = []
        seen_items = set()  # Track seen items to avoid duplicates
        
        # SIMPLE STRATEGY: Split numbered list and parse each item correctly
        # Format: "1. Item quantity unit 2. Item" or "1 Item 2 Item"
        
        # Step 1: Split text by numbered patterns: "1.", "2.", "1 ", "2 " etc.
        # Pattern matches: number followed by optional punctuation and space
        parts = re.split(r'(\d+[.,)]?\s+)', text)
        
        # Step 2: Process each part - odd indices are item texts, even indices are numbers
        item_texts = []
        for i in range(1, len(parts), 2):
            if i < len(parts):
                number = parts[i].strip()
                item_text = parts[i+1].strip() if i+1 < len(parts) else ""
                
                if len(item_text) > 1:
                    item_texts.append((number, item_text))
        
        # If no numbered pattern found, try alternative: look for standalone numbers
        if not item_texts:
            # Find all numbers and extract text after them
            for match in re.finditer(r'\d+[.,)]?\s+([A-Za-z].+?)(?=\s+\d+[.,)]?\s+[A-Za-z]|$)', text):
                item_text = match.group(1).strip()
                if len(item_text) > 1:
                    item_texts.append(("", item_text))
        
        if item_texts:
            print(f"📝 Processing {len(item_texts)} numbered items")
            
            for number, item_text in item_texts:
                if len(item_text) < 2:
                    continue
                
                # Clean OCR artifacts
                item_text = re.sub(r'[~=_|"\'"]+', ' ', item_text).strip()
                item_text = ' '.join(item_text.split())
                
                # Parse item: Look for patterns like:
                # - "Item quantity unit" (e.g., "Coconut 500g", "Milk 1L")
                # - "Item quantityunit" (e.g., "Coffee 250g")
                # - "Item" (no quantity/unit)
                
                item_name = ""
                quantity = "1"
                unit = "pack"
                
                # Pattern 1: Look for quantity+unit at the end: "Item 500g" or "Item 1L" or "Item 25_pcS"
                # Match: number followed by unit (g, kg, L, ml, pack, etc.)
                # Handle OCR artifacts like underscores: "100g_", "25_pcS", "Ikg" (I = 1)
                
                # First fix OCR errors: "Ikg" -> "1kg", "skg" -> "kg"
                item_text_fixed = re.sub(r'\bI([a-zA-Z])', r'1\1', item_text)  # "Ikg" -> "1kg"
                item_text_fixed = re.sub(r'\bs([a-zA-Z]{2,})', r'\1', item_text_fixed)  # "skg" -> "kg" (maybe)
                
                # Look for quantity+unit pattern: number + unit at end
                end_qty_unit = re.search(r'\s+(\d+(?:\.\d+)?)\s*[~=_]*\s*([a-zA-Z]{1,5})[~=_]*\s*$', item_text_fixed)
                if end_qty_unit:
                    quantity = end_qty_unit.group(1)
                    unit_raw = end_qty_unit.group(2).lower()
                    item_name = item_text_fixed[:end_qty_unit.start()].strip()
                    
                    # Clean and normalize unit (handle uppercase too)
                    unit = re.sub(r'^[lu_]', '', unit_raw.lower())
                    # Handle "pcS", "pcs", "pc" -> "pack"
                    if unit in ['pcs', 'pc', 'p', 'lpc', 'pcs']:
                        unit = 'pack'
                    elif unit in ['g', 'gm', 'kg', 'l', 'ml', 'litre', 'liter', 'og']:
                        if unit == 'gm' or unit == 'og':
                            unit = 'g'
                        elif unit == 'litre' or unit == 'liter':
                            unit = 'L'
                        # Keep g, kg, l, ml as is
                    elif len(unit) > 4 or not unit.isalpha():
                        unit = 'pack'  # Invalid unit
                    else:
                        # Check if it's a valid unit
                        valid_units = ['pack', 'bottle', 'can', 'jar', 'box', 'loaf', 'piece', 'pieces']
                        if unit not in valid_units:
                            unit = 'pack'
                else:
                    # Pattern 2: Look for quantity+unit in middle or at start
                    # Check if item starts with quantity+unit (like "500g Coconut")
                    start_qty_unit = re.match(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]{1,5})\s+(.+)', item_text)
                    if start_qty_unit:
                        quantity = start_qty_unit.group(1)
                        unit_raw = start_qty_unit.group(2).lower()
                        item_name = start_qty_unit.group(3).strip()
                        
                        # Clean unit
                        unit = re.sub(r'^[lu_]', '', unit_raw)
                        if unit in ['g', 'gm', 'kg', 'l', 'ml', 'og']:
                            if unit == 'gm' or unit == 'og':
                                unit = 'g'
                        elif unit == 'pcs' or unit == 'pc' or unit == 'lpc':
                            unit = 'pack'
                        else:
                            unit = 'pack'
                    else:
                        # Pattern 3: Look for any number+unit pattern (might be OCR error)
                        # Fix OCR errors first
                        item_text_fixed = re.sub(r'\bI([a-zA-Z])', r'1\1', item_text)
                        item_text_fixed = re.sub(r'\bs([a-zA-Z]{2,})', r'\1', item_text_fixed)
                        
                        all_qty_unit = re.findall(r'(\d+(?:\.\d+)?)\s*[~=_]*\s*([a-zA-Z]{1,4})', item_text_fixed)
                        if all_qty_unit:
                            # Use the last one (most likely to be quantity+unit)
                            quantity = all_qty_unit[-1][0]
                            unit_raw = all_qty_unit[-1][1].lower()
                            # Remove the matched pattern to get item name
                            pattern_to_remove = r'\s+' + re.escape(all_qty_unit[-1][0]) + r'\s*[~=_]*\s*' + re.escape(all_qty_unit[-1][1])
                            item_name = re.sub(pattern_to_remove, '', item_text_fixed).strip()
                            
                            # Clean unit
                            unit = re.sub(r'^[lu_]', '', unit_raw)
                            if unit in ['pcs', 'pc', 'p', 'lpc']:
                                unit = 'pack'
                            elif unit in ['g', 'gm', 'kg', 'l', 'ml', 'og']:
                                if unit == 'gm' or unit == 'og':
                                    unit = 'g'
                            else:
                                unit = 'pack'
                        else:
                            # No quantity/unit found - whole text is item name
                            item_name = item_text
                            quantity = "1"
                            unit = "pack"
                
                # Final cleaning of item name
                item_name = re.sub(r'^[~=_"]+|[~=_"]+$', '', item_name).strip()
                item_name = ' '.join(item_name.split())
                
                # Remove single letter artifacts
                words = item_name.split()
                item_name = ' '.join([w for w in words if len(w) > 1 or w.isalpha()])
                
                # Validate item name
                if len(item_name) > 1 and not item_name.lower() in ['pack', 'can', 'bottle', 'box']:
                    item_key = item_name.lower()
                    if item_key not in seen_items:
                        seen_items.add(item_key)
                        corrected_item_name = self.text_corrector.correct_item_name(item_name)
                        corrected_unit = self.text_corrector.correct_unit(unit)
                        
                        items.append({
                            'item_name': corrected_item_name,
                            'quantity': quantity,
                            'unit': corrected_unit,
                            'confidence': 0.8,
                            'original_item_name': item_name,
                            'original_unit': unit
                        })
        
        if items:
            print(f"✅ Extracted {len(items)} items from numbered list")
            return items
        
        # Strategy 1: Direct extraction from OCR text - split by patterns
        # Clean text but preserve structure
        cleaned_text = text
        
        # Strategy 1a: Split by common separators that indicate new items
        # Pattern: item ends with quantity+unit or special markers
        # Split on patterns like: "item ~U", "item =_U", "item _U", number followed by item name
        
        # First, try splitting on patterns that clearly separate items
        # Pattern: word followed by number or unit (end of item) followed by new word (start of item)
        split_points = []
        words = cleaned_text.split()
        
        # Find split points: transitions from quantity/unit to new item keyword
        for i in range(1, len(words)):
            prev_word = words[i-1].lower()
            curr_word = words[i].lower()
            
            # Previous word is unit or quantity
            is_prev_unit_or_qty = (prev_word in ['kg', 'g', 'gm', 'l', 'ml', 'pack', 'packs', 'bottle', 'can', 'jar', 'box', 'loaf', 'piece', 'pieces', 'head', 'each', 'litre', 'liter'] or 
                                  prev_word.isdigit() or
                                  re.match(r'^[lux]?\d+', prev_word))
            
            # Current word starts a new item
            is_curr_item = any(keyword in curr_word or curr_word.startswith(keyword[:3]) for keyword in self.common_items)
            
            if is_prev_unit_or_qty and is_curr_item:
                split_points.append(i)
        
        # Also split on special markers that separate items in OCR text
        # These patterns appear between items: "~U", "=_U", "_U", etc.
        split_patterns = [
            r'[~=_]+\s*[Uu]\s*',  # "~U", "=_U", "_U"
            r'\s+[~=_]+\s*[Uu]?\s+',  # Space followed by markers
            r'\s+(\d+)\s+(?=[A-Z][a-z])',  # Number followed by capital letter (new item)
        ]
        
        segments = [cleaned_text]
        for pattern in split_patterns:
            new_segments = []
            for segment in segments:
                parts = re.split(pattern, segment)
                new_segments.extend([p.strip() for p in parts if p.strip()])
            if len(new_segments) > len(segments):
                segments = new_segments
        
        # Extract items from each segment
        if len(segments) > 1:
            for segment in segments:
                if len(segment.strip()) > 3:
                    extracted = self._extract_from_text_segment(segment.strip(), seen_items)
                    items.extend(extracted)
            
            if items:
                return items
        
        # Strategy 1b: Handle OCR format with underscores and special chars
        # Clean text first - replace common OCR artifacts
        original_cleaned = cleaned_text
        cleaned_text = re.sub(r'[~=_]+', ' ', cleaned_text)  # Replace ~, =, _ with space
        cleaned_text = re.sub(r'[|]', ' ', cleaned_text)  # Replace | with space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single
        
        # Strategy 2: Extract items with patterns - handle OCR artifacts
        # Pattern examples: "Saffola_Oats 1L", "NutriChoice_ Digestive ~iscuits_ ~Upack"
        # Split text into segments where items likely are
        # Look for: ItemName followed by quantity+unit or unit marker patterns
        
        # Strategy 2: Handle OCR patterns with artifacts like "Lettuce_ 5 Lbead" or "Canned_tomatoes_ ~Lcan 65"
        # Clean up common OCR patterns: L/U before units, underscores, etc.
        # Pattern for items like "Lettuce 5 Lbead" -> "Lettuce 5 head" or "Canned_tomatoes Lcan 65" -> "Canned tomatoes can 65"
        ocr_patterns = [
            r'([A-Za-z]+(?:_[A-Za-z]+)*)\s+(\d+(?:\.\d+)?)\s*[LU]?([A-Za-z]+)',  # Handle L/U prefix on units
            r'([A-Za-z][A-Za-z\s]+?)\s+(?:[xX]\s*)?(\d+(?:\.\d+)?)\s+[LU]?([A-Za-z]+)',  # With x separator
            r'([A-Za-z]+(?:_[A-Za-z]+)*)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)',  # Standard pattern
        ]
        
        for pattern in ocr_patterns:
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            for match in matches:
                item_name = match[0].strip()
                quantity = match[1].strip()
                unit = match[2].strip().lower()
                
                # Clean item name - remove underscores and normalize
                item_name = re.sub(r'_', ' ', item_name).strip()
                item_name = ' '.join(item_name.split())
                
                # Clean unit - remove L/U prefix if present
                unit = re.sub(r'^[lu]', '', unit)
                
                if len(item_name) > 2 and len(unit) > 0:
                    # Skip if item name is just a unit name
                    if not item_name.lower() in ['pack', 'can', 'bottle', 'jar', 'loaf', 'box', 'head', 'slice']:
                        item_key = f"{item_name.lower()}_{quantity}_{unit}"
                        if item_key not in seen_items:
                            seen_items.add(item_key)
                            
                            corrected_item_name = self.text_corrector.correct_item_name(item_name)
                            corrected_unit = self.text_corrector.correct_unit(unit)
                            
                            items.append({
                                'item_name': corrected_item_name,
                                'quantity': quantity,
                                'unit': corrected_unit,
                                'confidence': 0.7,
                                'original_item_name': item_name,
                                'original_unit': unit
                            })
        
        # Strategy 3: Find patterns like "Item quantity unit" (standard format)
        # More flexible pattern that handles OCR errors
        if not items or len(items) < 3:  # Try additional patterns if we have few items
            patterns = [
                r'([A-Za-z][A-Za-z\s]+?)\s+(?:[xX]\s*)?(\d+(?:\.\d+)?)\s+([A-Za-z]+)',  # "Item 5 kg" or "Item x 5 kg"
                r'([A-Za-z][A-Za-z\s]+?)\s+(\d+(?:\.\d+)?)\s*([A-Za-z]+)',  # "Item 5kg"
                r'([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)',  # Multi-word items
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
                for match in matches:
                    item_name = match[0].strip()
                    quantity = match[1].strip()
                    unit = match[2].strip().lower()
                    
                    # Clean item name
                    item_name = re.sub(r'[_\-\W]+', ' ', item_name).strip()
                    item_name = ' '.join(item_name.split())  # Normalize whitespace
                    
                    # Clean unit - remove L/U prefix
                    unit = re.sub(r'^[lu]', '', unit)
                    
                    if len(item_name) > 2 and len(unit) > 0:
                        # Skip if item name is just a unit name
                        if not item_name.lower() in ['pack', 'can', 'bottle', 'jar', 'loaf', 'box', 'head', 'slice']:
                            # Create unique key to avoid duplicates
                            item_key = f"{item_name.lower()}_{quantity}_{unit}"
                            if item_key not in seen_items:
                                seen_items.add(item_key)
                                
                                corrected_item_name = self.text_corrector.correct_item_name(item_name)
                                corrected_unit = self.text_corrector.correct_unit(unit)
                                
                                items.append({
                                    'item_name': corrected_item_name,
                                    'quantity': quantity,
                                    'unit': corrected_unit,
                                    'confidence': 0.65,
                                    'original_item_name': item_name,
                                    'original_unit': unit
                                })
        
        # Strategy 3: Split by quantity+unit patterns if still no items
        if not items:
            parts = re.split(r'(\d+(?:\.\d+)?\s*[A-Za-z]+)', cleaned_text)
        
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
                        # Clean up item name (remove common separators and OCR artifacts)
                        item_name = re.sub(r'^[\-\s~=_LUI]+|[\-\s~=_LUI]+$', '', item_name)
                        item_name = re.sub(r'[~=_LUI]+', ' ', item_name)  # Replace artifacts with space
                        item_name = ' '.join(item_name.split())  # Normalize whitespace
                        
                        if len(item_name) > 2 and not item_name.lower() in ['pack', 'can', 'bottle', 'jar', 'loaf', 'box']:
                            # Create unique key to avoid duplicates
                            item_key = f"{item_name.lower()}_{quantity}_{unit}"
                            if item_key not in seen_items:
                                seen_items.add(item_key)
                                
                                # Apply text correction to item name and unit
                                corrected_item_name = self.text_corrector.correct_item_name(item_name)
                                corrected_unit = self.text_corrector.correct_unit(unit)
                                
                                items.append({
                                    'item_name': corrected_item_name,
                                    'quantity': quantity,
                                    'unit': corrected_unit,
                                    'confidence': 0.6,
                                    'original_item_name': item_name,
                                    'original_unit': unit
                                })
        
        # Strategy 4: Extract items even without explicit quantity/unit
        # For cases like "Saffola_Oats", "Patanjali_Honey", etc.
        if len(items) < 5:  # If we haven't found many items, be more aggressive
            # Look for capitalized words or known grocery keywords that might be item names
            # Pattern: Capitalized word(s) that could be brand/item names
            
            # Split text on special markers to find potential item boundaries
            segments = re.split(r'[~=_]+\s*[Uu]?\s*', text)
            
            for segment in segments:
                segment_clean = re.sub(r'[~=_|]+', ' ', segment).strip()
                segment_clean = ' '.join(segment_clean.split())
                
                if len(segment_clean) < 3:
                    continue
                
                # Check if segment contains known grocery keywords
                segment_lower = segment_clean.lower()
                has_keyword = any(keyword in segment_lower or segment_lower.startswith(keyword[:4]) 
                                 for keyword in self.common_items)
                
                # Try to extract quantity/unit
                qty_unit_match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]{1,5})\b', segment_clean)
                
                if qty_unit_match:
                    quantity = qty_unit_match.group(1)
                    unit = qty_unit_match.group(2).lower()
                    item_name = segment_clean[:qty_unit_match.start()].strip()
                    
                    # Clean item name
                    item_name = re.sub(r'^[~=_LUI]+\s*|\s*[~=_LUI]+$', '', item_name).strip()
                    item_name = ' '.join(item_name.split())
                    
                    if len(item_name) > 2:
                        item_key = item_name.lower()
                        if item_key not in seen_items:
                            seen_items.add(item_key)
                            corrected_item_name = self.text_corrector.correct_item_name(item_name)
                            corrected_unit = self.text_corrector.correct_unit(unit)
                            
                            items.append({
                                'item_name': corrected_item_name,
                                'quantity': quantity,
                                'unit': corrected_unit,
                                'confidence': 0.65,
                                'original_item_name': item_name,
                                'original_unit': unit
                            })
                elif has_keyword and len(segment_clean) > 5:
                    # Item name without quantity/unit - extract as item
                    item_name = segment_clean
                    item_name = re.sub(r'^[~=_LUI]+\s*|\s*[~=_LUI]+$', '', item_name).strip()
                    item_name = ' '.join(item_name.split())
                    
                    # Extract any trailing numbers as quantity
                    qty_match = re.search(r'\s+(\d+)\s*$', item_name)
                    if qty_match:
                        quantity = qty_match.group(1)
                        item_name = item_name[:qty_match.start()].strip()
                    else:
                        quantity = '1'
                    
                    if len(item_name) > 2:
                        item_key = item_name.lower()
                        if item_key not in seen_items:
                            seen_items.add(item_key)
                            corrected_item_name = self.text_corrector.correct_item_name(item_name)
                            
                            items.append({
                                'item_name': corrected_item_name,
                                'quantity': quantity,
                                'unit': 'pack',
                                'confidence': 0.6,
                                'original_item_name': item_name,
                                'original_unit': 'pack'
                            })
        
        # Strategy 5: Line-by-line parsing for lists without numbers
        if not items:
            lines = text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 3]
            
            if len(non_empty_lines) > 1:
                print(f"📝 Processing {len(non_empty_lines)} lines as potential items")
                for line in non_empty_lines:
                    if any(word in line.lower() for word in ['grocery', 'items', 'extract', 'ocr text', 'item_name']):
                        continue
                    
                    qty_unit_match = re.search(r'(\d+(?:\.\d+)?)\s+([a-zA-Z]+)', line)
                    if qty_unit_match:
                        quantity = qty_unit_match.group(1)
                        unit = qty_unit_match.group(2).lower()
                        item_name = line[:qty_unit_match.start()].strip()
                    else:
                        item_name = line
                        quantity = '1'
                        unit = 'pack'
                    
                    item_name = re.sub(r'[~=_|]+', ' ', item_name).strip()
                    item_name = ' '.join(item_name.split())
                    
                    if len(item_name) > 2:
                        item_key = item_name.lower()
                        if item_key not in seen_items:
                            seen_items.add(item_key)
                            corrected_item_name = self.text_corrector.correct_item_name(item_name)
                            corrected_unit = self.text_corrector.correct_unit(unit)
                            
                            items.append({
                                'item_name': corrected_item_name,
                                'quantity': quantity,
                                'unit': corrected_unit,
                                'confidence': 0.6,
                                'original_item_name': item_name,
                                'original_unit': unit
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
