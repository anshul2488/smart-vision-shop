#!/usr/bin/env python3
"""
Optimized Flask Backend for Grocery OCR Pipeline
Integrates OCR model, scraper, and provides API endpoints
Features: Lazy Loading, Model Caching, Uvicorn Support
"""

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import os
import sys
import json
import base64
import io
from PIL import Image
import pandas as pd
import torch
import cv2
import numpy as np
from datetime import datetime
import logging
import threading
import time
from functools import lru_cache
from typing import Optional, Dict, Any, List
import re
import difflib

# Try to import pyspellchecker
try:
    from spellchecker import SpellChecker
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'smartshop_vision_secret_key_2024'  # For session management
CORS(app)

# Configure logging - show info and above for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model Cache and Lazy Loading
class ModelCache:
    """Thread-safe model cache with lazy loading"""
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.RLock()
        self._initialization_status = {
            'ocr_processor': False,
            'llm_processor': False,
            'model': False,
            'amazon_scraper': False,
            'preprocessor': False,
            'handwritten_ocr': False
        }
        self._initialization_errors = {}
    
    def get_or_load(self, component_name: str, loader_func, *args, **kwargs):
        """Get component from cache or load it lazily"""
        with self._lock:
            if component_name not in self._cache:
                try:
                    logger.info(f"🔄 Lazy loading {component_name}...")
                    start_time = time.time()
                    self._cache[component_name] = loader_func(*args, **kwargs)
                    load_time = time.time() - start_time
                    self._initialization_status[component_name] = True
                    logger.info(f"✅ {component_name} loaded in {load_time:.2f}s")
                except Exception as e:
                    logger.error(f"❌ Failed to load {component_name}: {e}")
                    self._initialization_errors[component_name] = str(e)
                    return None
            return self._cache[component_name]
    
    def is_loaded(self, component_name: str) -> bool:
        """Check if component is loaded"""
        return self._initialization_status.get(component_name, False)
    
    def get_status(self) -> Dict[str, Any]:
        """Get initialization status of all components"""
        return {
            'status': self._initialization_status,
            'errors': self._initialization_errors,
            'cache_size': len(self._cache)
        }

# Global model cache instance
model_cache = ModelCache()

# Fast Text Corrector (spell checker) - optimized with caching
class FastTextCorrector:
    """Optimized text corrector with caching and batch processing"""
    
    def __init__(self):
        # Common grocery item corrections (OCR errors -> correct)
        self.item_corrections = {
            # Common misspellings
            'basmai': 'basmati', 'basmti': 'basmati', 'basmat': 'basmati',
            'grigina': 'original', 'orginal': 'original', 'orginl': 'original', 'orignal': 'original',
            'lpack': 'pack', 'pak': 'pack', 'pck': 'pack',
            'pousdec': 'powder', 'pousedec': 'powder', 'powdr': 'powder', 'powde': 'powder',
            'chilli': 'chili', 'chili': 'chili', 'chilies': 'chili',
            'redchilli': 'red chili', 'redchili': 'red chili',
            'tamatar': 'tomato', 'tamato': 'tomato', 'tamatr': 'tomato',
            'pyaz': 'onion', 'pyaaz': 'onion',
            'aloo': 'potato', 'alo': 'potato',
            'dahi': 'curd', 'doodh': 'milk', 'dudh': 'milk',
            'makhan': 'butter', 'chawal': 'rice', 'gehun': 'wheat',
            'atta': 'flour', 'chini': 'sugar', 'namak': 'salt',
            'nimbu': 'lemon', 'adrak': 'ginger', 'lehsun': 'garlic',
            'paneer': 'cheese', 'chai': 'tea', 'kapi': 'coffee',
            'namkeen': 'chips', 'sabun': 'soap', 'shampo': 'shampoo',
            # Common product name patterns
            'bisciut': 'biscuit', 'biscuts': 'biscuit', 'biscut': 'biscuit',
            'biscuits': 'biscuit', 'biscket': 'biscuit',
            'pacle': 'parle', 'parleg': 'parle-g', 'pacle-g': 'parle-g',
            # Additional common OCR errors
            'tto': 'to', 'sapann': 'soap', 'looc': 'loaf',
        }
        
        # Unit corrections (common OCR errors)
        self.unit_corrections = {
            'sun': 'kg', 'kehirooc': 'kg', 'kgrooc': 'kg', 'kgirooc': 'kg',
            'kilrooc': 'kg', 'kilgrooc': 'kg', 'q': 'kg',
            'k': 'kg', 'kg': 'kg', 'kilogram': 'kg', 'kilo': 'kg',
            'g': 'g', 'gram': 'g', 'gm': 'g', 'grams': 'g',
            'l': 'l', 'liter': 'l', 'litre': 'l',
            'ml': 'ml', 'milliliter': 'ml', 'millilitre': 'ml',
            'pack': 'pack', 'packs': 'pack', 'pkt': 'pack', 'packet': 'pack',
            'pcs': 'pieces', 'pc': 'pieces', 'piece': 'pieces', 'pieces': 'pieces',
            'dozen': 'dozen', 'dz': 'dozen',
            'box': 'box', 'boxes': 'box',
            'bottle': 'bottle', 'bottles': 'bottle',
            'tin': 'tin', 'tins': 'tin', 'can': 'can', 'cans': 'can',
            'bag': 'bag', 'bags': 'bag', 'loaf': 'loaf', 'loaves': 'loaf',
        }
        
        # Cache for corrections
        self.correction_cache = {}
        
        # Initialize spell checker if available
        self.spell_checker = None
        if SPELLCHECK_AVAILABLE:
            try:
                self.spell_checker = SpellChecker()
                # Add grocery terms to dictionary
                grocery_terms = list(self.item_corrections.values())
                for term in grocery_terms:
                    if isinstance(term, str) and len(term.split()) == 1:
                        self.spell_checker.word_frequency.load_words([term])
            except Exception as e:
                logger.warning(f"Spell checker initialization failed: {e}")
                self.spell_checker = None
    
    def correct_item_name(self, item_name: str) -> str:
        """Correct grocery item name with spell checking (cached)"""
        if not item_name or len(item_name.strip()) < 2:
            return item_name
        
        # Check cache first
        cache_key = item_name.lower().strip()
        if cache_key in self.correction_cache:
            # Preserve original case if cached
            cached = self.correction_cache[cache_key]
            if item_name.isupper():
                return cached.upper()
            elif item_name.istitle():
                return cached.title()
            return cached
        
        original = item_name.strip()
        # Normalize: replace underscores and special chars with spaces
        normalized = re.sub(r'[_\-\W]+', ' ', original)
        corrected = normalized.lower().strip()
        
        # First, check direct corrections (exact match)
        if corrected in self.item_corrections:
            result = self.item_corrections[corrected].title()
            self.correction_cache[cache_key] = result
            return result
        
        # Check for partial matches in multi-word items
        words = corrected.split()
        corrected_words = []
        has_correction = False
        
        for word in words:
            original_word = word
            # Clean word (remove special chars)
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
        
        # If correction is too different, return original
        similarity = difflib.SequenceMatcher(None, original.lower(), corrected_name.lower()).ratio()
        if similarity < 0.5 and not has_correction:
            self.correction_cache[cache_key] = original
            return original
        
        self.correction_cache[cache_key] = corrected_name
        return corrected_name
    
    def correct_unit(self, unit: str) -> str:
        """Correct unit name"""
        if not unit:
            return unit
        
        unit_lower = unit.lower().strip()
        
        # Check cache
        if unit_lower in self.correction_cache:
            return self.correction_cache[unit_lower]
        
        # Direct match
        if unit_lower in self.unit_corrections:
            result = self.unit_corrections[unit_lower]
            self.correction_cache[unit_lower] = result
            return result
        
        # Fuzzy match for similar units
        best_match = self._fuzzy_match_word(unit_lower, list(self.unit_corrections.keys()))
        if best_match:
            similarity = difflib.SequenceMatcher(None, unit_lower, best_match).ratio()
            if similarity > 0.6:
                result = self.unit_corrections[best_match]
                self.correction_cache[unit_lower] = result
                return result
        
        # Default to original if no good match
        self.correction_cache[unit_lower] = unit
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
    
    def correct_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Correct a single item dictionary"""
        corrected_item = item.copy()
        
        # Correct item name
        if 'item_name' in item:
            original_name = item['item_name']
            corrected_name = self.correct_item_name(original_name)
            corrected_item['item_name'] = corrected_name
            if original_name != corrected_name:
                corrected_item['original_item_name'] = original_name
        
        # Correct unit
        if 'unit' in item:
            original_unit = item['unit']
            corrected_unit = self.correct_unit(original_unit)
            corrected_item['unit'] = corrected_unit
            if original_unit != corrected_unit:
                corrected_item['original_unit'] = original_unit
        
        # Update search query if it exists
        if 'search_query' in corrected_item:
            corrected_item['search_query'] = f"{corrected_item.get('item_name', '')} {corrected_item.get('quantity', '')} {corrected_item.get('unit', '')}".strip()
        
        return corrected_item
    
    def correct_text(self, text: str) -> str:
        """Correct entire text line"""
        if not text:
            return text
        
        # Try to extract item pattern: "item_name [x] quantity unit"
        pattern = r'^(.+?)\s+(?:x\s+)?(\d+(?:\.\d+)?)\s+([a-zA-Z]+)$'
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        
        if match:
            item_name = match.group(1).strip()
            quantity = match.group(2)
            unit = match.group(3).strip()
            
            corrected_name = self.correct_item_name(item_name)
            corrected_unit = self.correct_unit(unit)
            
            return f"{corrected_name} {quantity} {corrected_unit}"
        else:
            # Just correct words in text
            words = text.split()
            corrected_words = []
            
            for word in words:
                # Check if it looks like a unit
                if word.lower() in self.unit_corrections or self._fuzzy_match_word(word.lower(), list(self.unit_corrections.keys())):
                    corrected_words.append(self.correct_unit(word))
                else:
                    corrected_words.append(self.correct_item_name(word))
            
            return ' '.join(corrected_words)

# Global spell checker instance (lazy loaded)
_spell_checker_instance = None

def get_spell_checker():
    """Get or create global spell checker instance"""
    global _spell_checker_instance
    if _spell_checker_instance is None:
        _spell_checker_instance = FastTextCorrector()
    return _spell_checker_instance

# Pincode management
PINCODE_FILE = "user_pincode.json"

def load_saved_pincode():
    """Load saved pincode from file."""
    try:
        if os.path.exists(PINCODE_FILE):
            with open(PINCODE_FILE, 'r') as f:
                data = json.load(f)
                return data.get('pincode')
    except Exception as e:
        logger.warning(f"Could not load saved pincode: {e}")
    return None

def save_pincode(pincode: str):
    """Save pincode to file for future use."""
    try:
        data = {'pincode': pincode, 'updated_at': datetime.now().isoformat()}
        with open(PINCODE_FILE, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved pincode: {pincode}")
        return True
    except Exception as e:
        logger.warning(f"Could not save pincode: {e}")
        return False

def is_valid_pincode(pincode: str):
    """Validate pincode format."""
    return pincode and len(pincode) == 6 and pincode.isdigit()

def get_user_pincode():
    """Get pincode from session or file."""
    # First check session
    if 'pincode' in session and is_valid_pincode(session['pincode']):
        return session['pincode']
    
    # Then check saved file
    saved_pincode = load_saved_pincode()
    if saved_pincode and is_valid_pincode(saved_pincode):
        session['pincode'] = saved_pincode
        return saved_pincode
    
    return None

def set_user_pincode(pincode: str):
    """Set pincode in session and save to file."""
    if is_valid_pincode(pincode):
        session['pincode'] = pincode
        save_pincode(pincode)
        
        # Note: New Blinkit scraper doesn't require pincode setting
        # The pincode is handled through session headers
        
        return True
    return False

# Lazy loading functions
def load_ocr_processor():
    """Lazy load OCR processor"""
    from grocery_ocr_llm_model import OCRProcessor
    return OCRProcessor()

def load_llm_processor():
    """Lazy load LLM processor"""
    try:
        from grocery_ocr_llm_model import LLMProcessor
        processor = LLMProcessor()
        logger.info("✅ LLMProcessor initialized successfully (LLM disabled, direct extraction only)")
        return processor
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLMProcessor: {e}", exc_info=True)
        # Return None - the calling code will handle fallback
        return None

def load_grocery_model():
    """Lazy load grocery OCR model (DEPRECATED: Use handwritten OCR instead)"""
    from grocery_ocr_llm_model import GroceryOCRModel
    
    ocr_processor = model_cache.get_or_load('ocr_processor', load_ocr_processor)
    llm_processor = model_cache.get_or_load('llm_processor', load_llm_processor)
    
    if not ocr_processor or not llm_processor:
        raise Exception("Failed to load OCR or LLM processor")
    
    model = GroceryOCRModel(ocr_processor, llm_processor)
    
    # Try to load trained handwritten OCR model (best_model.pth) first
    # This is the trained CRNN model from train_handwritten_ocr.py
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            # Note: This won't work directly as GroceryOCRModel doesn't match HandwrittenOCRModel architecture
            # But we try to load if compatible
            if 'model_state_dict' in checkpoint:
                # Only load if architectures match (they probably don't, but try)
                try:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    logger.info("✅ Loaded best_model.pth (partial match)")
                except:
                    logger.warning("⚠️ best_model.pth architecture doesn't match GroceryOCRModel, using untrained model")
            else:
                logger.warning("⚠️ best_model.pth doesn't contain model_state_dict")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load best_model.pth: {e}")
    else:
        logger.warning(f"⚠️ best_model.pth not found at {model_path}, using untrained model")
    
    model.eval()
    return model

def load_amazon_scraper():
    """Lazy load Amazon scraper"""
    from Scrapper.amazon_scraper import AmazonScraper
    return AmazonScraper()

def load_preprocessor():
    """Lazy load preprocessor"""
    from specialized_preprocessing import GroceryListPreprocessor
    return GroceryListPreprocessor()

def load_handwritten_ocr():
    """
    Lazy load handwritten OCR model.
    This uses models/best_model.pth from train_handwritten_ocr.py training.
    This is the PRIMARY OCR method used in parse_image() endpoint.
    """
    from handwritten_ocr_integration import HandwrittenOCRIntegration
    
    # Explicitly use best_model.pth from training (not grocery_ocr_model.pth)
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ best_model.pth not found at {model_path}. Handwritten OCR will not be available.")
        logger.info("💡 Train the model first: python train_handwritten_ocr.py")
    
    # Create instance with explicit path to best_model.pth
    handwritten_ocr = HandwrittenOCRIntegration(model_path=model_path)
    logger.info(f"✅ Handwritten OCR loaded from: {model_path}")
    return handwritten_ocr

# Cached property decorators for lazy loading
@lru_cache(maxsize=1)
def get_ocr_processor():
    return model_cache.get_or_load('ocr_processor', load_ocr_processor)

@lru_cache(maxsize=1)
def get_llm_processor():
    return model_cache.get_or_load('llm_processor', load_llm_processor)

@lru_cache(maxsize=1)
def get_grocery_model():
    return model_cache.get_or_load('model', load_grocery_model)

@lru_cache(maxsize=1)
def get_amazon_scraper():
    return model_cache.get_or_load('amazon_scraper', load_amazon_scraper)

@lru_cache(maxsize=1)
def get_preprocessor():
    return model_cache.get_or_load('preprocessor', load_preprocessor)

@lru_cache(maxsize=1)
def get_handwritten_ocr():
    return model_cache.get_or_load('handwritten_ocr', load_handwritten_ocr)

# Removed old initialization - now using lazy loading

@app.route('/')
def index():
    """Serve the main frontend page"""
    return send_from_directory('Frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from Frontend directory"""
    return send_from_directory('Frontend', filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with lazy loading status"""
    cache_status = model_cache.get_status()
    
    # Check if handwritten OCR is available (lazy load)
    handwritten_ocr_available = False
    handwritten_ocr_info = {}
    try:
        handwritten_ocr = get_handwritten_ocr()
        if handwritten_ocr:
            handwritten_ocr_available = handwritten_ocr.is_available()
            handwritten_ocr_info = handwritten_ocr.get_model_info()
    except Exception as e:
        logger.warning(f"Handwritten OCR not available: {e}")
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'lazy_loading': True,
        'components': cache_status['status'],
        'initialization_errors': cache_status['errors'],
        'cache_size': cache_status['cache_size'],
        'handwritten_ocr_available': handwritten_ocr_available,
        'handwritten_ocr_info': handwritten_ocr_info,
        'pincode': get_user_pincode()
    })

@app.route('/api/pincode', methods=['GET'])
def get_pincode():
    """Get current pincode"""
    try:
        pincode = get_user_pincode()
        return jsonify({
            'pincode': pincode,
            'is_set': pincode is not None,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/pincode', methods=['POST'])
def set_pincode_endpoint():
    """Set pincode for delivery location"""
    try:
        data = request.get_json()
        if not data or 'pincode' not in data:
            return jsonify({
                'error': 'Pincode is required',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        pincode = str(data['pincode']).strip()
        
        if not is_valid_pincode(pincode):
            return jsonify({
                'error': 'Invalid pincode. Please enter a 6-digit pincode.',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        if set_user_pincode(pincode):
            return jsonify({
                'success': True,
                'pincode': pincode,
                'message': 'Pincode saved successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': 'Failed to save pincode',
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/parse-image', methods=['POST'])
def parse_image():
    """Parse grocery list from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Save temporary image for processing
        temp_path = f"temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_path, image_cv)
        
        try:
            # Try handwritten OCR first if available (lazy load)
            best_text = ""
            best_confidence = 0
            ocr_method = "fallback"
            handwritten_ocr = None  # Initialize variable
            
            try:
                handwritten_ocr = get_handwritten_ocr()
                if handwritten_ocr and handwritten_ocr.is_available():
                    logger.info("Using trained handwritten OCR model...")
                try:
                    # Use hybrid approach (EasyOCR + CRNN)
                    best_text, best_confidence = handwritten_ocr.predict_hybrid(temp_path)
                    ocr_method = "handwritten_hybrid"
                    logger.info(f"Handwritten OCR result: confidence={best_confidence:.3f}, text_length={len(best_text)}")
                    
                    # Create synthetic OCR results from lines for spatial processing
                    lines = best_text.split('\n')
                    ocr_results = []
                    for i, line in enumerate(lines):
                        if line.strip():
                            ocr_results.append((line.strip(), best_confidence, (0, i*30, 100, 30)))
                except Exception as e:
                    logger.warning(f"Handwritten OCR failed: {e}")
                    # Fallback to CRNN only
                    try:
                        best_text = handwritten_ocr.predict_crnn(temp_path)
                        best_confidence = 0.6  # Default confidence for CRNN
                        ocr_method = "handwritten_crnn"
                        logger.info(f"CRNN-only result: text_length={len(best_text)}")
                        
                        # Create synthetic OCR results from lines
                        lines = best_text.split('\n')
                        ocr_results = []
                        for i, line in enumerate(lines):
                            if line.strip():
                                ocr_results.append((line.strip(), best_confidence, (0, i*30, 100, 30)))
                    except Exception as e2:
                        logger.warning(f"CRNN-only also failed: {e2}")
                        ocr_results = []
            except Exception as e:
                logger.warning(f"Handwritten OCR not available: {e}")
                handwritten_ocr = None
                ocr_results = []
            
            # Initialize ocr_results variable if not set
            if 'ocr_results' not in locals():
                ocr_results = []
            
            # If handwritten OCR didn't work or not available, use traditional OCR
            if not best_text.strip():
                logger.info("Using traditional OCR with preprocessing...")
                # Use specialized preprocessing for better OCR results (lazy load)
                preprocessor = get_preprocessor()
                processed_versions = preprocessor.preprocess_grocery_list(temp_path, "temp_preprocessed")
                
                # Test different preprocessed versions
                test_versions = [
                    ("Final", "06_final.jpg"),
                    ("Inverted", "07_inverted.jpg"), 
                    ("High_Contrast", "08_high_contrast.jpg"),
                    ("Enhanced", "02_enhanced.jpg")
                ]
                
                for version_name, filename in test_versions:
                    try:
                        version_path = os.path.join("temp_preprocessed", filename)
                        if os.path.exists(version_path):
                            # Process with OCR directly on the preprocessed file (lazy load)
                            ocr_processor = get_ocr_processor()
                            processed_image = ocr_processor.preprocess_image_optimal(version_path)
                            version_ocr_results = ocr_processor.extract_text_optimal(processed_image)
                            
                            if version_ocr_results:
                                # Calculate average confidence
                                avg_confidence = sum(conf for _, conf, _ in version_ocr_results) / len(version_ocr_results)
                                
                                # Combine text
                                predicted_text = ' '.join(text for text, _, _ in version_ocr_results)
                                
                                # If this version has better confidence, use it
                                if avg_confidence > best_confidence and len(predicted_text.strip()) > len(best_text.strip()):
                                    best_confidence = avg_confidence
                                    best_text = predicted_text
                                    ocr_results = version_ocr_results  # Save OCR results with spatial info
                                    ocr_method = f"traditional_{version_name.lower()}"
                                    
                                    logger.info(f"Better OCR result from {version_name}: confidence={avg_confidence:.3f}, text_length={len(predicted_text)}")
                    
                    except Exception as e:
                        logger.warning(f"Error processing {version_name}: {e}")
                        continue
                
                # If no good results from preprocessed versions, fall back to original
                if not best_text.strip():
                    logger.info("Falling back to original image processing")
                    ocr_processor = get_ocr_processor()
                    processed_image = ocr_processor.preprocess_image_optimal(temp_path)
                    ocr_results = ocr_processor.extract_text_optimal(processed_image)
                    best_text = ' '.join(text for text, _, _ in ocr_results)
                    best_confidence = sum(conf for _, conf, _ in ocr_results) / len(ocr_results) if ocr_results else 0
                    ocr_method = "traditional_original"
            
            # Process text extraction (NO LLM dependency)
            llm_processor = get_llm_processor()
            
            # If LLM processor failed to load, use direct extraction
            if llm_processor is None:
                logger.warning("LLM processor not available, using direct extraction")
                from grocery_ocr_llm_model import LLMProcessor
                try:
                    # Create a simple processor just for extraction
                    simple_processor = LLMProcessor()
                    llm_results = simple_processor.process_text(best_text)
                except Exception as e:
                    logger.error(f"Failed to create processor: {e}")
                    # Final fallback: create minimal result
                    from grocery_ocr_llm_model import LLMProcessor
                    temp_processor = LLMProcessor()
                    llm_results = temp_processor._extract_items_simple(best_text)
                    llm_results = {
                        'original_text': best_text,
                        'llm_output': 'Direct extraction (processor init failed)',
                        'extracted_items': llm_results if isinstance(llm_results, list) else [],
                        'confidence': 0.6
                    }
            else:
                llm_results = llm_processor.process_text(best_text)
            
            # Additional processing: Use OCR spatial information if available
            if ocr_results and len(llm_results.get('extracted_items', [])) < 2:
                # Try to use spatial grouping from OCR results
                logger.info("Using spatial information from OCR to improve extraction")
                # Group OCR results by position
                spatial_items = []
                seen_positions = set()
                
                for text, conf, bbox in ocr_results:
                    y_pos = bbox[1]  # y coordinate
                    # Group by similar y position (same line)
                    position_key = (y_pos // 30) * 30  # Round to nearest 30 pixels
                    
                    if position_key not in seen_positions:
                        seen_positions.add(position_key)
                        # Try to extract items from this line
                        if llm_processor:
                            line_items = llm_processor._extract_items_simple(text)
                        else:
                            from grocery_ocr_llm_model import LLMProcessor
                            temp_proc = LLMProcessor()
                            line_items = temp_proc._extract_items_simple(text)
                        spatial_items.extend(line_items)
                
                if spatial_items:
                    logger.info(f"Spatial extraction found {len(spatial_items)} items")
                    # Merge with LLM results, avoiding duplicates
                    existing_names = {item.get('item_name', '').lower() for item in llm_results.get('extracted_items', [])}
                    for item in spatial_items:
                        if item.get('item_name', '').lower() not in existing_names:
                            llm_results['extracted_items'].append(item)
                            existing_names.add(item.get('item_name', '').lower())
            
            # Extract items and apply spell checking
            spell_checker = get_spell_checker()
            items = []
            
            # Ensure extracted_items exists and is a list
            extracted_items = llm_results.get('extracted_items', [])
            if not isinstance(extracted_items, list):
                extracted_items = []
            
            logger.info(f"LLM extracted {len(extracted_items)} items from text: {best_text[:100]}")
            logger.info(f"LLM results: {llm_results}")
            
            for item in extracted_items:
                try:
                    # Apply spell checking corrections
                    corrected_item = spell_checker.correct_item({
                        'item_name': item.get('item_name', ''),
                        'quantity': item.get('quantity', ''),
                        'unit': item.get('unit', '')
                    })
                    # Include both 'name' and 'item_name' for frontend compatibility
                    item_name = corrected_item['item_name']
                    items.append({
                        'name': item_name,  # Frontend expects 'name'
                        'item_name': item_name,  # Also include 'item_name' for compatibility
                        'quantity': str(corrected_item.get('quantity', item.get('quantity', ''))),
                        'unit': corrected_item['unit'] or item.get('unit', ''),
                        'confidence': item.get('confidence', 0.6),
                        'original_item_name': corrected_item.get('original_item_name'),
                        'original_unit': corrected_item.get('original_unit')
                    })
                except Exception as e:
                    logger.warning(f"Error correcting item {item}: {e}")
                    # Fallback: add item without correction
                    item_name = item.get('item_name', '')
                    items.append({
                        'name': item_name,  # Frontend expects 'name'
                        'item_name': item_name,  # Also include 'item_name' for compatibility
                        'quantity': str(item.get('quantity', '')),
                        'unit': item.get('unit', ''),
                        'confidence': item.get('confidence', 0.6)
                    })
            
            logger.info(f"Final items count: {len(items)}")
            logger.info(f"Items: {items}")
            
            # Clean up temp files
            os.remove(temp_path)
            import shutil
            if os.path.exists("temp_preprocessed"):
                shutil.rmtree("temp_preprocessed")
            
            response_data = {
                'success': True,
                'raw_text': best_text,
                'items': items,
                'total_items': len(items),
                'confidence': llm_results.get('confidence', 0.6),
                'ocr_confidence': best_confidence,
                'processing_info': {
                    'ocr_method': ocr_method,
                    'handwritten_ocr_available': handwritten_ocr.is_available() if handwritten_ocr else False,
                    'preprocessing_used': ocr_method.startswith('traditional'),
                    'best_confidence': best_confidence
                }
            }
            
            logger.info(f"Sending response with {len(items)} items")
            return jsonify(response_data)
            
        except Exception as e:
            # Clean up temp files on error
            logger.error(f"Error parsing image: {e}", exc_info=True)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            if os.path.exists("temp_preprocessed"):
                try:
                    import shutil
                    shutil.rmtree("temp_preprocessed")
                except:
                    pass
            raise e
            
    except Exception as e:
        logger.error(f"Error parsing image: {e}")
        return jsonify({'error': f'Failed to parse image: {str(e)}'}), 500

@app.route('/api/parse-text', methods=['POST'])
def parse_text():
    """Parse grocery list from text input"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Process text extraction (NO LLM dependency)
        llm_processor = get_llm_processor()
        
        # If LLM processor failed to load, use direct extraction
        if llm_processor is None:
            logger.warning("LLM processor not available, using direct extraction")
            from grocery_ocr_llm_model import LLMProcessor
            try:
                simple_processor = LLMProcessor()
                llm_results = simple_processor.process_text(text)
            except Exception as e:
                logger.error(f"Failed to create processor: {e}")
                # Final fallback
                from grocery_ocr_llm_model import LLMProcessor
                temp_processor = LLMProcessor()
                direct_items = temp_processor._extract_items_simple(text)
                llm_results = {
                    'original_text': text,
                    'llm_output': 'Direct extraction (processor init failed)',
                    'extracted_items': direct_items if isinstance(direct_items, list) else [],
                    'confidence': 0.6
                }
        else:
            llm_results = llm_processor.process_text(text)
        
        # Extract items and apply spell checking
        spell_checker = get_spell_checker()
        items = []
        extracted_items = llm_results.get('extracted_items', [])
        if not isinstance(extracted_items, list):
            extracted_items = []
        
        for item in extracted_items:
            try:
                # Apply spell checking corrections
                corrected_item = spell_checker.correct_item({
                    'item_name': item.get('item_name', ''),
                    'quantity': item.get('quantity', ''),
                    'unit': item.get('unit', '')
                })
                # Include both 'name' and 'item_name' for frontend compatibility
                item_name = corrected_item['item_name']
                items.append({
                    'name': item_name,  # Frontend expects 'name'
                    'item_name': item_name,  # Also include 'item_name' for compatibility
                    'quantity': str(corrected_item.get('quantity', item.get('quantity', ''))),
                    'unit': corrected_item['unit'] or item.get('unit', ''),
                    'confidence': item.get('confidence', 0.6),
                    'original_item_name': corrected_item.get('original_item_name'),
                    'original_unit': corrected_item.get('original_unit')
                })
            except Exception as e:
                logger.warning(f"Error correcting item {item}: {e}")
                # Fallback: add item without correction
                item_name = item.get('item_name', '')
                items.append({
                    'name': item_name,  # Frontend expects 'name'
                    'item_name': item_name,  # Also include 'item_name' for compatibility
                    'quantity': str(item.get('quantity', '')),
                    'unit': item.get('unit', ''),
                    'confidence': item.get('confidence', 0.6)
                })
        
        return jsonify({
            'success': True,
            'raw_text': text,
            'items': items,
            'total_items': len(items),
            'confidence': llm_results['confidence']
        })
        
    except Exception as e:
        logger.error(f"Error parsing text: {e}")
        return jsonify({'error': f'Failed to parse text: {str(e)}'}), 500

@app.route('/api/search-prices', methods=['POST'])
def search_prices():
    """Search prices for grocery items from multiple platforms"""
    try:
        data = request.get_json()
        items = data.get('items', [])
        platforms = data.get('platforms', ['amazon', 'zepto', 'bigbasket', 'jiomart'])  # Default to all platforms
        
        if not items:
            return jsonify({'error': 'No items provided'}), 400
        
        results = []
        
        # Use fast spell checker for correction before search
        text_corrector = get_spell_checker()
        
        for item in items:
            item_name = item.get('item_name', '')
            quantity = item.get('quantity', '')
            unit = item.get('unit', '')
            
            if not item_name:
                continue
            
            # Apply additional text correction before search (safeguard)
            corrected_item_name = text_corrector.correct_item_name(item_name)
            corrected_unit = text_corrector.correct_unit(unit)
            
            # Use corrected values for search
            search_query = f"{corrected_item_name} {quantity} {corrected_unit}".strip()
            
            item_results = {
                'item_name': corrected_item_name,
                'quantity': quantity,
                'unit': corrected_unit,
                'search_query': search_query,
                'original_item_name': item_name,  # Keep original for display
                'original_unit': unit,  # Keep original for display
                'platforms': {}
            }
            
            # Use the price scraper for smart matching
            from price_scraper import PriceScraper
            price_scraper = PriceScraper()
            
            # Create user item dictionary (use corrected values)
            user_item = {
                'item_name': corrected_item_name,
                'quantity': quantity,
                'unit': corrected_unit
            }
            
            # Get best prices with smart matching (without saving individual files)
            best_prices = price_scraper.get_best_prices(user_item, max_results=5)
            
            # Debug logging
            logger.info(f"Best prices result for {item_name}: platforms={list(best_prices['all_products'].keys())}")
            logger.info(f"BigBasket products count: {len(best_prices['all_products'].get('bigbasket', []))}")
            
            # Process results for each platform - only include platforms with products
            for platform in platforms:
                logger.info(f"Processing platform: {platform}")
                if platform in best_prices['all_products']:
                    # Get all products for this platform
                    all_platform_products = best_prices['all_products'][platform]
                    logger.info(f"Platform {platform} has {len(all_platform_products)} products")
                    
                    # Only add platform if it has products
                    if all_platform_products and len(all_platform_products) > 0:
                        platform_products = []
                        
                        for product in all_platform_products:
                            # Calculate price for this product
                            try:
                                raw_price = float(product.get('price', 0))
                            except (ValueError, TypeError):
                                raw_price = 0.0
                            
                            # For now, use raw price as calculated_total to avoid calculation errors
                            # TODO: Implement proper price calculation logic
                            calculated_total = raw_price
                            
                            platform_products.append({
                                'name': product.get('name', ''),
                                'price': product.get('price', ''),
                                'unit_price': raw_price,
                                'calculated_total': calculated_total,
                                'rating': product.get('rating', ''),
                                'review_count': product.get('review_count', ''),
                                'product_url': product.get('product_url', ''),
                                'image_url': product.get('image_url', ''),
                                'brand': product.get('brand', ''),
                                'variant': product.get('variant', ''),
                                'inventory': product.get('inventory', 0),
                                'eta': product.get('eta', ''),
                                'match_score': 10,  # Default match score
                                'total_products_found': len(all_platform_products)
                            })
                        
                        # Only add platform if we have at least one product with valid data
                        if platform_products:
                            item_results['platforms'][platform] = {
                                'products': platform_products
                            }
                    # else: Platform has no products, don't include it
                # else: Platform not in results, don't include it
            
            results.append(item_results)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_items': len(results),
            'platforms_searched': platforms
        })
        
    except Exception as e:
        logger.error(f"Error searching prices: {e}")
        return jsonify({'error': f'Failed to search prices: {str(e)}'}), 500

@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample grocery list data"""
    sample_items = [
        {'item_name': 'Tomatoes', 'quantity': '1', 'unit': 'kg'},
        {'item_name': 'Bread', 'quantity': '2', 'unit': 'loaves'},
        {'item_name': 'Milk', 'quantity': '1', 'unit': 'L'},
        {'item_name': 'Eggs', 'quantity': '12', 'unit': 'pieces'},
        {'item_name': 'Rice', 'quantity': '5', 'unit': 'kg'}
    ]
    
    return jsonify({
        'success': True,
        'items': sample_items,
        'total_items': len(sample_items)
    })

@app.route('/api/test-handwritten-ocr', methods=['POST'])
def test_handwritten_ocr():
    """Test handwritten OCR model specifically"""
    try:
        # Lazy load handwritten OCR
        handwritten_ocr = get_handwritten_ocr()
        if not handwritten_ocr or not handwritten_ocr.is_available():
            return jsonify({'error': 'Handwritten OCR model not available'}), 400
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Save temporary image for processing
        temp_path = f"temp_handwritten_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_path, image_cv)
        
        try:
            # Test different OCR methods
            results = {}
            
            # CRNN only
            try:
                crnn_text = handwritten_ocr.predict_crnn(temp_path)
                results['crnn_only'] = {
                    'text': crnn_text,
                    'confidence': 0.6,
                    'method': 'CRNN'
                }
            except Exception as e:
                results['crnn_only'] = {'error': str(e)}
            
            # Hybrid (EasyOCR + CRNN)
            try:
                hybrid_text, hybrid_confidence = handwritten_ocr.predict_hybrid(temp_path)
                results['hybrid'] = {
                    'text': hybrid_text,
                    'confidence': hybrid_confidence,
                    'method': 'EasyOCR + CRNN'
                }
            except Exception as e:
                results['hybrid'] = {'error': str(e)}
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'results': results,
                'model_info': handwritten_ocr.get_model_info()
            })
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error testing handwritten OCR: {e}")
        return jsonify({'error': f'Failed to test handwritten OCR: {str(e)}'}), 500

@app.route('/api/preprocessing-info', methods=['GET'])
def get_preprocessing_info():
    """Get information about preprocessing techniques used"""
    return jsonify({
        'success': True,
        'preprocessing_techniques': [
            {
                'name': 'Line Removal',
                'description': 'Removes blue lines and red margins from lined paper',
                'technique': 'HSV color space filtering'
            },
            {
                'name': 'Handwriting Enhancement',
                'description': 'Bilateral filtering, CLAHE, and gamma correction',
                'technique': 'Multi-step image enhancement'
            },
            {
                'name': 'Adaptive Thresholding',
                'description': 'Combines Gaussian, Mean, and Otsu thresholding',
                'technique': 'Multi-method thresholding'
            },
            {
                'name': 'Noise Removal',
                'description': 'Morphological operations and connected component analysis',
                'technique': 'Morphological filtering'
            },
            {
                'name': 'Deskewing',
                'description': 'Automatic rotation correction based on text lines',
                'technique': 'PCA-based angle detection'
            },
            {
                'name': 'Final Enhancement',
                'description': 'Contrast and sharpness enhancement for OCR',
                'technique': 'PIL-based enhancement'
            }
        ],
        'ocr_versions_tested': [
            'Final (recommended)',
            'Inverted (for dark text on light)',
            'High Contrast (for low contrast)',
            'Enhanced (basic enhancement)'
        ],
        'confidence_scoring': 'Uses average confidence across all detected text regions',
        'handwritten_ocr_available': get_handwritten_ocr().is_available() if get_handwritten_ocr() else False,
        'handwritten_ocr_info': get_handwritten_ocr().get_model_info() if get_handwritten_ocr() else {}
    })

@app.route('/api/spell-check', methods=['POST'])
def spell_check():
    """Spell check and correct text/items"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        spell_checker = get_spell_checker()
        
        # Handle single text string
        if 'text' in data:
            text = data.get('text', '')
            corrected = spell_checker.correct_text(text)
            return jsonify({
                'success': True,
                'original': text,
                'corrected': corrected
            })
        
        # Handle single item
        if 'item' in data:
            item = data.get('item', {})
            corrected_item = spell_checker.correct_item(item)
            return jsonify({
                'success': True,
                'original': item,
                'corrected': corrected_item
            })
        
        # Handle list of items
        if 'items' in data:
            items = data.get('items', [])
            corrected_items = []
            for item in items:
                corrected_item = spell_checker.correct_item(item)
                corrected_items.append(corrected_item)
            
            return jsonify({
                'success': True,
                'original': items,
                'corrected': corrected_items,
                'total_items': len(corrected_items)
            })
        
        return jsonify({'error': 'Invalid request. Provide "text", "item", or "items"'}), 400
        
    except Exception as e:
        logger.error(f"Error in spell check: {e}")
        return jsonify({'error': f'Failed to spell check: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Create ASGI application for Uvicorn
try:
    from asgiref.wsgi import WsgiToAsgi
    asgi_app = WsgiToAsgi(app)
except ImportError:
    asgi_app = None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Grocery OCR Pipeline Server')
    parser.add_argument('--server', choices=['flask', 'uvicorn'], default='uvicorn',
                       help='Server type to use (default: uvicorn)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes (uvicorn only)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload (development only)')
    
    args = parser.parse_args()
    
    if args.server == 'uvicorn':
        try:
            import uvicorn
            
            logger.info("🚀 Starting server with Uvicorn (optimized)")
            logger.info(f"📍 Server: {args.server}")
            logger.info(f"🌐 Host: {args.host}:{args.port}")
            logger.info(f"👥 Workers: {args.workers}")
            logger.info("⚡ Features: Lazy Loading, Model Caching, ASGI")
            
            uvicorn.run(
                "app:asgi_app",
                host=args.host,
                port=args.port,
                workers=args.workers,
                reload=args.reload,
                access_log=True,
                log_level="info"
            )
        except ImportError:
            logger.warning("Uvicorn not available, falling back to Flask development server")
            app.run(debug=args.reload, host=args.host, port=args.port)
    else:
        logger.info("🚀 Starting server with Flask development server")
        logger.info("⚠️  For production, use: python app.py --server uvicorn")
        app.run(debug=args.reload, host=args.host, port=args.port)
