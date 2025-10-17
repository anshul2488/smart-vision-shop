#!/usr/bin/env python3

import cv2
import numpy as np
import paddleocr
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter
import re
import difflib
from typing import List, Tuple, Dict
import logging
from dataclasses import dataclass
import json
import os
import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from collections import defaultdict

@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    method: str

@dataclass
class TrainingResult:
    image_path: str
    ground_truth: str
    predicted: str
    accuracy: float
    method: str

class HandwrittenOCRTrainer:
    
    def __init__(self, dataset_path: str = r"C:\Users\Anshul Shinde\Desktop\SEM 7\BTECH\dataset\train_data"):
        self.dataset_path = dataset_path
        self.grocery_vocabulary = self._load_grocery_vocabulary()
        self.training_results = []
        self.performance_metrics = {}
        
        # Initialize logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize PaddleOCR models
        self.paddle_ocr_reader = None
        self.paddle_ocr_handwritten = None
        self._initialize_paddle_ocr()
    
    def _initialize_paddle_ocr(self):
        """Initialize PaddleOCR models"""
        self.logger.info("Initializing PaddleOCR models...")
        
        try:
            # Initialize standard PaddleOCR for general text
            self.paddle_ocr_reader = PaddleOCR(
                use_angle_cls=True, 
                lang='en',
                use_gpu=False,  # Set to True if you have GPU
                show_log=False
            )
            self.logger.info("PaddleOCR standard model initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize PaddleOCR standard model: {e}")
            self.paddle_ocr_reader = None
        
        try:
            # Initialize PaddleOCR with handwritten text optimization
            self.paddle_ocr_handwritten = PaddleOCR(
                use_angle_cls=True, 
                lang='en',
                use_gpu=False,  # Set to True if you have GPU
                show_log=False,
                det_model_dir=None,  # Use default detection model
                rec_model_dir=None,  # Use default recognition model
                cls_model_dir=None,  # Use default classification model
                det_limit_side_len=960,  # Increase for better detection
                det_limit_type='max',
                rec_batch_num=6,  # Batch size for recognition
                max_text_length=25,  # Max text length
                rec_image_shape="3, 32, 320",  # Image shape for recognition
                use_space_char=True,  # Use space character
                drop_score=0.3,  # Lower threshold for handwritten text
                use_mp=False,  # Disable multiprocessing for stability
                total_process_num=1
            )
            self.logger.info("PaddleOCR handwritten model initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize PaddleOCR handwritten model: {e}")
            self.paddle_ocr_handwritten = None
        
        self.logger.info("PaddleOCR model initialization completed.")
        
    def _load_grocery_vocabulary(self) -> List[str]:
        """Load comprehensive grocery vocabulary"""
        return [
            # Dairy & Proteins
            'milk', 'butter', 'ghee', 'cheese', 'paneer', 'curd', 'yogurt', 'cream',
            'eggs', 'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'turkey',
            
            # Grains & Staples
            'rice', 'basmati', 'wheat', 'flour', 'maida', 'bread', 'pasta', 'noodles',
            'sugar', 'salt', 'oil', 'ghee', 'vinegar', 'sauce',
            
            # Vegetables & Fruits
            'onions', 'tomatoes', 'potatoes', 'carrots', 'cabbage', 'cauliflower',
            'spinach', 'coriander', 'mint', 'ginger', 'garlic', 'lemons',
            'apples', 'bananas', 'oranges', 'grapes', 'mangoes',
            
            # Spices & Seasonings
            'turmeric', 'chili', 'pepper', 'cumin', 'coriander', 'garam', 'masala',
            'tea', 'coffee', 'cocoa', 'vanilla',
            
            # Packaged Foods
            'maggi', 'lays', 'oreo', 'biscuits', 'chips', 'coke', 'pepsi',
            'detergent', 'soap', 'shampoo', 'toothpaste',
            
            # Quantities
            'kg', 'gm', 'litre', 'ml', 'pack', 'bottle', 'bunch', 'dozen'
        ]
    
    def load_dataset(self) -> List[Tuple[str, str]]:
        """Load all image-text pairs from the dataset"""
        image_text_pairs = []
        
        # Get all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.dataset_path, ext)))
        
        self.logger.info(f"Found {len(image_files)} image files")
        
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
                self.logger.info(f"Loaded: {os.path.basename(image_path)} -> {len(ground_truth)} chars")
            else:
                self.logger.warning(f"No text file found for: {image_path}")
        
        self.logger.info(f"Successfully loaded {len(image_text_pairs)} image-text pairs")
        return image_text_pairs
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Advanced image preprocessing for handwritten text"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.logger.info(f"Original image shape: {image.shape}")
        
        # Step 1: Resize image if too small or too large
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            # Upscale small images
            scale_factor = max(100/height, 100/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            self.logger.info(f"Upscaled image to: {image.shape}")
        elif height > 2000 or width > 2000:
            # Downscale very large images
            scale_factor = min(2000/height, 2000/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            self.logger.info(f"Downscaled image to: {image.shape}")
        
        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Noise reduction with multiple techniques
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Additional denoising with Non-local Means
        denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
        
        # Step 4: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Step 5: Gamma correction for better contrast
        gamma = 1.5
        gamma_corrected = np.power(enhanced / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(gamma_corrected)
        
        # Step 6: Multiple thresholding approaches
        # Otsu's thresholding
        _, otsu_thresh = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gamma_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Combine both thresholding methods
        combined_thresh = cv2.bitwise_and(otsu_thresh, adaptive_thresh)
        
        # Step 7: Morphological operations to clean up
        # Remove small noise
        kernel_noise = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_noise)
        
        # Fill small holes
        kernel_fill = np.ones((3, 3), np.uint8)
        filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_fill)
        
        # Step 8: Deskewing (text rotation correction)
        filled = self._deskew_image(filled)
        
        # Step 9: Final cleanup
        # Remove very small connected components (noise)
        filled = self._remove_small_components(filled, min_size=50)
        
        # Step 10: Edge enhancement for better text clarity
        kernel_edge = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced_edges = cv2.filter2D(filled, -1, kernel_edge)
        
        # Ensure binary output
        _, final_result = cv2.threshold(enhanced_edges, 127, 255, cv2.THRESH_BINARY)
        
        self.logger.info(f"Preprocessing completed. Final shape: {final_result.shape}")
        return final_result
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in the image"""
        try:
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Correct angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Only rotate if angle is significant
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                       flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
                self.logger.info(f"Deskewed image by {angle:.2f} degrees")
                return rotated
            
        except Exception as e:
            self.logger.warning(f"Deskewing failed: {e}")
        
        return image
    
    def _remove_small_components(self, image: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Remove small connected components (noise)"""
        try:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
            
            # Create mask for components larger than min_size
            mask = np.zeros_like(image)
            for i in range(1, num_labels):  # Skip background (label 0)
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    mask[labels == i] = 255
            
            return mask
            
        except Exception as e:
            self.logger.warning(f"Component removal failed: {e}")
            return image
    
    def extract_text_paddleocr(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using PaddleOCR with optimized parameters for handwritten text"""
        results = []
        
        if not self.paddle_ocr_reader and not self.paddle_ocr_handwritten:
            self.logger.warning("No PaddleOCR models available")
            return results
        
        try:
            # Try both PaddleOCR models
            ocr_models = []
            if self.paddle_ocr_reader:
                ocr_models.append(('PaddleOCR-Standard', self.paddle_ocr_reader))
            if self.paddle_ocr_handwritten:
                ocr_models.append(('PaddleOCR-Handwritten', self.paddle_ocr_handwritten))
            
            best_results = []
            best_confidence = 0
            
            for model_name, ocr_model in ocr_models:
                try:
                    # PaddleOCR returns results in format: [[[bbox], (text, confidence)], ...]
                    detections = ocr_model.ocr(image, cls=True)
                    
                    if detections and detections[0]:
                        # Calculate average confidence for this model
                        confidences = [det[1][1] for det in detections[0]]
                        avg_confidence = sum(confidences) / len(confidences)
                        
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_results = (model_name, detections[0])
                            
                except Exception as e:
                    self.logger.warning(f"PaddleOCR {model_name} failed: {e}")
                    continue
            
            # Use the best results
            if best_results:
                model_name, detections = best_results
                for detection in detections:
                    bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text, confidence = detection[1]
                    
                    # Convert bbox to (x, y, w, h) format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    w, h = x2 - x1, y2 - y1
                    
                    # Filter out very low confidence results
                    if confidence > 0.1:  # Lower threshold for handwritten text
                        results.append(OCRResult(
                            text=text.strip(),
                            confidence=confidence,
                            bbox=(int(x1), int(y1), int(w), int(h)),
                            method=model_name
                        ))
            
            # If no good results, try with even more lenient parameters
            if not results and self.paddle_ocr_handwritten:
                try:
                    # Try with different image preprocessing
                    enhanced_image = self._enhance_for_paddleocr(image)
                    detections = self.paddle_ocr_handwritten.ocr(enhanced_image, cls=True)
                    
                    if detections and detections[0]:
                        for detection in detections[0]:
                            bbox = detection[0]
                            text, confidence = detection[1]
                            
                            # Convert bbox to (x, y, w, h) format
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            w, h = x2 - x1, y2 - y1
                            
                            results.append(OCRResult(
                                text=text.strip(),
                                confidence=confidence,
                                bbox=(int(x1), int(y1), int(w), int(h)),
                                method='PaddleOCR-Enhanced'
                            ))
                except Exception as e:
                    self.logger.warning(f"Enhanced PaddleOCR also failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"PaddleOCR extraction failed: {e}")
        
        return results
    
    def _enhance_for_paddleocr(self, image: np.ndarray) -> np.ndarray:
        """Enhance image specifically for PaddleOCR"""
        try:
            # Convert to PIL for enhancement
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Convert back to numpy
            enhanced = np.array(pil_image)
            if len(enhanced.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def extract_text_paddleocr_alternative(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using alternative PaddleOCR configuration"""
        results = []
        
        if not self.paddle_ocr_reader:
            self.logger.warning("No PaddleOCR models available for alternative extraction")
            return results
        
        try:
            # Try with different preprocessing approaches
            preprocessing_methods = [
                ("Original", image),
                ("Enhanced", self._enhance_for_paddleocr(image)),
                ("Inverted", self._invert_image(image)),
                ("High_Contrast", self._high_contrast_preprocessing(image))
            ]
            
            best_results = []
            best_confidence = 0
            
            for method_name, processed_image in preprocessing_methods:
                try:
                    # Use standard PaddleOCR with different preprocessing
                    detections = self.paddle_ocr_reader.ocr(processed_image, cls=True)
                    
                    if detections and detections[0]:
                        # Calculate average confidence
                        confidences = [det[1][1] for det in detections[0]]
                        avg_confidence = sum(confidences) / len(confidences)
                        
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_results = (method_name, detections[0])
                            
                except Exception as e:
                    self.logger.warning(f"PaddleOCR alternative {method_name} failed: {e}")
                    continue
            
            # Use the best results
            if best_results:
                method_name, detections = best_results
                for detection in detections:
                    bbox = detection[0]
                    text, confidence = detection[1]
                    
                    # Convert bbox to (x, y, w, h) format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    w, h = x2 - x1, y2 - y1
                    
                    # Filter out very low confidence results
                    if confidence > 0.1:
                        results.append(OCRResult(
                            text=text.strip(),
                            confidence=confidence,
                            bbox=(int(x1), int(y1), int(w), int(h)),
                            method=f'PaddleOCR-{method_name}'
                        ))
                    
        except Exception as e:
            self.logger.error(f"PaddleOCR alternative extraction failed: {e}")
        
        return results
    
    def _invert_image(self, image: np.ndarray) -> np.ndarray:
        """Invert image colors"""
        try:
            return 255 - image
        except Exception as e:
            self.logger.warning(f"Image inversion failed: {e}")
            return image
    
    def _high_contrast_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply high contrast preprocessing"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            return thresh
            
        except Exception as e:
            self.logger.warning(f"High contrast preprocessing failed: {e}")
            return image
    
    def post_process_text(self, ocr_results: List[OCRResult]) -> str:
        """Post-process OCR results into clean text with enhanced handwritten text handling"""
        processed_text = []
        
        for result in ocr_results:
            text = result.text.strip()
            
            # More lenient filtering for handwritten text
            if len(text) < 1 or result.confidence < 0.05:  # Lower confidence threshold
                continue
            
            # Apply multiple rounds of corrections
            text = self._apply_ocr_corrections(text)
            text = self._apply_handwritten_corrections(text)
            text = self._apply_advanced_corrections(text)
            
            # Correct with vocabulary
            corrected = self._correct_with_vocabulary(text)
            
            if corrected and len(corrected) > 0:
                processed_text.append(corrected)
        
        # Join with newlines and clean up
        final_text = '\n'.join(processed_text)
        
        # Additional cleanup for handwritten text
        final_text = self._clean_handwritten_text(final_text)
        
        return final_text
    
    def _apply_handwritten_corrections(self, text: str) -> str:
        """Apply corrections specific to handwritten text"""
        handwritten_corrections = {
            # Common handwritten OCR errors
            'rn': 'm', 'cl': 'd', 'vv': 'w', 'uu': 'w',
            '0': 'o', '1': 'l', '5': 's', '8': 'b', '6': 'g', '9': 'g',
            'rnilk': 'milk', 'brecid': 'bread', 'egqs': 'eggs',
            'checise': 'cheese', 'tornatoes': 'tomatoes', 'orticins': 'onions',
            'carrcits': 'carrots', 'chickcin': 'chicken', 'ghe': 'ghee',
            'paneer': 'paneer', 'basrnati': 'basmati', 'rnaggi': 'maggi',
            'rnays': 'lays', 'oreo': 'oreo', 'butter': 'butter',
            'sugar': 'sugar', 'wheat': 'wheat', 'flour': 'flour',
            'rice': 'rice', 'tea': 'tea', 'turmeric': 'turmeric',
            'chili': 'chili', 'onions': 'onions', 'potatoes': 'potatoes',
            'tomatoes': 'tomatoes', 'coriander': 'coriander', 'eggs': 'eggs',
            'coke': 'coke', 'fresh': 'fresh', 'bunch': 'bunch',
            'pack': 'pack', 'bottle': 'bottle', 'litre': 'litre',
            'kg': 'kg', 'gm': 'gm', 'ml': 'ml', 'g': 'g',
            # Quantity corrections
            '1/2': '1/2', '1/4': '1/4', '3/4': '3/4',
            '1/2kg': '1/2 kg', '1/4kg': '1/4 kg', '3/4kg': '3/4 kg',
            '1/2litre': '1/2 litre', '1/4litre': '1/4 litre', '3/4litre': '3/4 litre',
        }
        
        for error, correction in handwritten_corrections.items():
            text = text.replace(error, correction)
        
        return text
    
    def _apply_advanced_corrections(self, text: str) -> str:
        """Apply advanced corrections using pattern matching"""
        import re
        
        # Fix spacing around quantities
        text = re.sub(r'(\d+)(kg|gm|litre|ml|g|l)', r'\1 \2', text)
        text = re.sub(r'(\d+/\d+)(kg|gm|litre|ml|g|l)', r'\1 \2', text)
        
        # Fix spacing around dashes
        text = re.sub(r'(\w)-(\w)', r'\1 - \2', text)
        
        # Fix common OCR spacing issues
        text = re.sub(r'(\w)(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)(\w)', r'\1 \2', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _clean_handwritten_text(self, text: str) -> str:
        """Clean up handwritten text output"""
        import re
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Remove lines that are too short (likely noise)
        cleaned_lines = []
        for line in lines:
            if len(line) >= 2:  # Keep lines with at least 2 characters
                cleaned_lines.append(line)
        
        # Join lines and clean up
        result = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        result = re.sub(r'\n\s*\n', '\n', result)
        
        return result.strip()
    
    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR corrections for handwritten text"""
        corrections = {
            'rn': 'm',
            'cl': 'd',
            'vv': 'w',
            '0': 'o',
            '1': 'l',
            '5': 's',
            '8': 'b',
            'rnilk': 'milk',
            'brecid': 'bread',
            'egqs': 'eggs',
            'checise': 'cheese',
            'tornatoes': 'tomatoes',
            'orticins': 'onions',
            'carrcits': 'carrots',
            'chickcin': 'chicken',
            'ghe': 'ghee',
            'paneer': 'paneer',
            'basrnati': 'basmati',
            'rnaggi': 'maggi',
            'rnays': 'lays',
            'oreo': 'oreo',
            'rnilk': 'milk',
            'butter': 'butter',
            'sugar': 'sugar',
            'wheat': 'wheat',
            'flour': 'flour',
            'rice': 'rice',
            'tea': 'tea',
            'turmeric': 'turmeric',
            'chili': 'chili',
            'onions': 'onions',
            'potatoes': 'potatoes',
            'tomatoes': 'tomatoes',
            'coriander': 'coriander',
            'eggs': 'eggs',
            'coke': 'coke',
            'fresh': 'fresh',
            'bunch': 'bunch',
            'pack': 'pack',
            'bottle': 'bottle',
            'litre': 'litre',
            'kg': 'kg',
            'gm': 'gm',
            'ml': 'ml'
        }
        
        for error, correction in corrections.items():
            text = text.replace(error, correction)
        
        return text
    
    def _correct_with_vocabulary(self, text: str) -> str:
        """Correct text using grocery vocabulary"""
        text_lower = text.lower()
        
        if text_lower in self.grocery_vocabulary:
            return text
        
        # Find close matches
        matches = difflib.get_close_matches(text_lower, self.grocery_vocabulary, n=1, cutoff=0.6)
        if matches:
            return matches[0]
        
        # Check for partial matches
        for vocab_word in self.grocery_vocabulary:
            if text_lower in vocab_word or vocab_word in text_lower:
                similarity = difflib.SequenceMatcher(None, text_lower, vocab_word).ratio()
                if similarity > 0.5:
                    return vocab_word
        
        # Return original text if it's reasonable
        if len(text) > 1 and (text.isalpha() or any(c.isdigit() for c in text)):
            return text
        
        return None
    
    def calculate_accuracy(self, ground_truth: str, predicted: str) -> float:
        """Calculate accuracy between ground truth and predicted text"""
        # Normalize text
        gt_normalized = self._normalize_text(ground_truth)
        pred_normalized = self._normalize_text(predicted)
        
        # Calculate character-level accuracy
        if len(gt_normalized) == 0:
            return 1.0 if len(pred_normalized) == 0 else 0.0
        
        # Use sequence matcher for better accuracy calculation
        matcher = difflib.SequenceMatcher(None, gt_normalized, pred_normalized)
        return matcher.ratio()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,/\-\(\)]', '', text)
        
        return text.strip()
    
    def train_and_evaluate(self) -> Dict:
        """Train and evaluate OCR on the dataset"""
        self.logger.info("Loading dataset...")
        dataset = self.load_dataset()
        
        if not dataset:
            self.logger.error("No dataset found!")
            return {}
        
        self.logger.info(f"Training on {len(dataset)} samples...")
        
        paddleocr_results = []
        paddleocr_alt_results = []
        
        for i, (image_path, ground_truth) in enumerate(dataset):
            self.logger.info(f"Processing {i+1}/{len(dataset)}: {os.path.basename(image_path)}")
            
            try:
                # Preprocess image
                processed_image = self.preprocess_image(image_path)
                
                # Extract text with PaddleOCR
                paddleocr_detections = self.extract_text_paddleocr(processed_image)
                paddleocr_text = self.post_process_text(paddleocr_detections)
                paddleocr_accuracy = self.calculate_accuracy(ground_truth, paddleocr_text)
                
                paddleocr_results.append(TrainingResult(
                    image_path=image_path,
                    ground_truth=ground_truth,
                    predicted=paddleocr_text,
                    accuracy=paddleocr_accuracy,
                    method='PaddleOCR'
                ))
                
                # Extract text with PaddleOCR Alternative
                paddleocr_alt_detections = self.extract_text_paddleocr_alternative(processed_image)
                paddleocr_alt_text = self.post_process_text(paddleocr_alt_detections)
                paddleocr_alt_accuracy = self.calculate_accuracy(ground_truth, paddleocr_alt_text)
                
                paddleocr_alt_results.append(TrainingResult(
                    image_path=image_path,
                    ground_truth=ground_truth,
                    predicted=paddleocr_alt_text,
                    accuracy=paddleocr_alt_accuracy,
                    method='PaddleOCR-Alternative'
                ))
                
                self.logger.info(f"PaddleOCR accuracy: {paddleocr_accuracy:.3f}")
                self.logger.info(f"PaddleOCR Alternative accuracy: {paddleocr_alt_accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                continue
        
        # Calculate overall metrics
        self.performance_metrics = self._calculate_metrics(paddleocr_results, paddleocr_alt_results)
        
        # Save results
        self._save_training_results(paddleocr_results, paddleocr_alt_results)
        
        return self.performance_metrics
    
    def _calculate_metrics(self, paddleocr_results: List[TrainingResult], 
                          paddleocr_alt_results: List[TrainingResult]) -> Dict:
        """Calculate performance metrics"""
        metrics = {}
        
        # PaddleOCR metrics
        paddleocr_accuracies = [r.accuracy for r in paddleocr_results]
        metrics['paddleocr'] = {
            'mean_accuracy': np.mean(paddleocr_accuracies),
            'std_accuracy': np.std(paddleocr_accuracies),
            'min_accuracy': np.min(paddleocr_accuracies),
            'max_accuracy': np.max(paddleocr_accuracies),
            'total_samples': len(paddleocr_results)
        }
        
        # PaddleOCR Alternative metrics
        paddleocr_alt_accuracies = [r.accuracy for r in paddleocr_alt_results]
        metrics['paddleocr_alternative'] = {
            'mean_accuracy': np.mean(paddleocr_alt_accuracies),
            'std_accuracy': np.std(paddleocr_alt_accuracies),
            'min_accuracy': np.min(paddleocr_alt_accuracies),
            'max_accuracy': np.max(paddleocr_alt_accuracies),
            'total_samples': len(paddleocr_alt_results)
        }
        
        # Combined metrics
        all_accuracies = paddleocr_accuracies + paddleocr_alt_accuracies
        metrics['overall'] = {
            'mean_accuracy': np.mean(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'total_samples': len(all_accuracies)
        }
        
        return metrics
    
    def _save_training_results(self, paddleocr_results: List[TrainingResult], 
                              paddleocr_alt_results: List[TrainingResult]):
        """Save training results to files"""
        # Save detailed results
        results_data = {
            'paddleocr_results': [
                {
                    'image_path': r.image_path,
                    'ground_truth': r.ground_truth,
                    'predicted': r.predicted,
                    'accuracy': r.accuracy
                } for r in paddleocr_results
            ],
            'paddleocr_alternative_results': [
                {
                    'image_path': r.image_path,
                    'ground_truth': r.ground_truth,
                    'predicted': r.predicted,
                    'accuracy': r.accuracy
                } for r in paddleocr_alt_results
            ],
            'performance_metrics': self.performance_metrics
        }
        
        with open('ocr_training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary = {
            'training_summary': {
                'total_samples': len(paddleocr_results),
                'paddleocr_mean_accuracy': self.performance_metrics['paddleocr']['mean_accuracy'],
                'paddleocr_alternative_mean_accuracy': self.performance_metrics['paddleocr_alternative']['mean_accuracy'],
                'best_method': 'PaddleOCR' if self.performance_metrics['paddleocr']['mean_accuracy'] > 
                              self.performance_metrics['paddleocr_alternative']['mean_accuracy'] else 'PaddleOCR-Alternative'
            }
        }
        
        with open('ocr_training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Training results saved to ocr_training_results.json")
        self.logger.info("Training summary saved to ocr_training_summary.json")
    
    def display_results(self):
        """Display training results"""
        if not self.performance_metrics:
            self.logger.error("No training results available. Run train_and_evaluate() first.")
            return
        
        print("\n" + "="*60)
        print("OCR TRAINING RESULTS")
        print("="*60)
        
        print(f"\nDataset: {self.dataset_path}")
        print(f"Total samples: {self.performance_metrics['overall']['total_samples']}")
        
        print(f"\nPaddleOCR Performance:")
        print(f"  Mean Accuracy: {self.performance_metrics['paddleocr']['mean_accuracy']:.3f}")
        print(f"  Std Deviation: {self.performance_metrics['paddleocr']['std_accuracy']:.3f}")
        print(f"  Min Accuracy:  {self.performance_metrics['paddleocr']['min_accuracy']:.3f}")
        print(f"  Max Accuracy:  {self.performance_metrics['paddleocr']['max_accuracy']:.3f}")
        
        print(f"\nPaddleOCR Alternative Performance:")
        print(f"  Mean Accuracy: {self.performance_metrics['paddleocr_alternative']['mean_accuracy']:.3f}")
        print(f"  Std Deviation: {self.performance_metrics['paddleocr_alternative']['std_accuracy']:.3f}")
        print(f"  Min Accuracy:  {self.performance_metrics['paddleocr_alternative']['min_accuracy']:.3f}")
        print(f"  Max Accuracy:  {self.performance_metrics['paddleocr_alternative']['max_accuracy']:.3f}")
        
        best_method = 'PaddleOCR' if self.performance_metrics['paddleocr']['mean_accuracy'] > \
                     self.performance_metrics['paddleocr_alternative']['mean_accuracy'] else 'PaddleOCR-Alternative'
        
        print(f"\nBest Method: {best_method}")
        print(f"Overall Mean Accuracy: {self.performance_metrics['overall']['mean_accuracy']:.3f}")

def main():
    """Main function to run OCR training"""
    print("Handwritten OCR Training System")
    print("="*40)
    
    # Initialize trainer
    trainer = HandwrittenOCRTrainer()
    
    # Train and evaluate
    print("Starting training and evaluation...")
    metrics = trainer.train_and_evaluate()
    
    if metrics:
        trainer.display_results()
        print("\nTraining completed successfully!")
        print("Check ocr_training_results.json for detailed results.")
    else:
        print("Training failed. Check the logs for errors.")

if __name__ == "__main__":
    main()
