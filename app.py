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
from typing import Optional, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'smartshop_vision_secret_key_2024'  # For session management
CORS(app)

# Configure logging - only show errors
logging.basicConfig(level=logging.ERROR)
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
    from grocery_ocr_llm_model import LLMProcessor
    return LLMProcessor()

def load_grocery_model():
    """Lazy load grocery OCR model"""
    from grocery_ocr_llm_model import GroceryOCRModel
    
    ocr_processor = model_cache.get_or_load('ocr_processor', load_ocr_processor)
    llm_processor = model_cache.get_or_load('llm_processor', load_llm_processor)
    
    if not ocr_processor or not llm_processor:
        raise Exception("Failed to load OCR or LLM processor")
    
    model = GroceryOCRModel(ocr_processor, llm_processor)
    
    # Load trained model if available
    model_path = "grocery_ocr_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Loaded trained model")
    else:
        logger.warning("⚠️ No trained model found, using untrained model")
    
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
    """Lazy load handwritten OCR"""
    from handwritten_ocr_integration import handwritten_ocr
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
            
            try:
                handwritten_ocr = get_handwritten_ocr()
                if handwritten_ocr and handwritten_ocr.is_available():
                    logger.info("Using trained handwritten OCR model...")
                try:
                    # Use hybrid approach (EasyOCR + CRNN)
                    best_text, best_confidence = handwritten_ocr.predict_hybrid(temp_path)
                    ocr_method = "handwritten_hybrid"
                    logger.info(f"Handwritten OCR result: confidence={best_confidence:.3f}, text_length={len(best_text)}")
                except Exception as e:
                    logger.warning(f"Handwritten OCR failed: {e}")
                    # Fallback to CRNN only
                    try:
                        best_text = handwritten_ocr.predict_crnn(temp_path)
                        best_confidence = 0.6  # Default confidence for CRNN
                        ocr_method = "handwritten_crnn"
                        logger.info(f"CRNN-only result: text_length={len(best_text)}")
                    except Exception as e2:
                        logger.warning(f"CRNN-only also failed: {e2}")
            except Exception as e:
                logger.warning(f"Handwritten OCR not available: {e}")
            
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
                            ocr_results = ocr_processor.extract_text_optimal(processed_image)
                            
                            if ocr_results:
                                # Calculate average confidence
                                avg_confidence = sum(conf for _, conf, _ in ocr_results) / len(ocr_results)
                                
                                # Combine text
                                predicted_text = ' '.join(text for text, _, _ in ocr_results)
                                
                                # If this version has better confidence, use it
                                if avg_confidence > best_confidence and len(predicted_text.strip()) > len(best_text.strip()):
                                    best_confidence = avg_confidence
                                    best_text = predicted_text
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
            
            # Process with LLM (lazy load)
            llm_processor = get_llm_processor()
            llm_results = llm_processor.process_text(best_text)
            
            # Extract items
            items = []
            for item in llm_results['extracted_items']:
                items.append({
                    'item_name': item['item_name'],
                    'quantity': item['quantity'],
                    'unit': item['unit'],
                    'confidence': item['confidence']
                })
            
            # Clean up temp files
            os.remove(temp_path)
            import shutil
            if os.path.exists("temp_preprocessed"):
                shutil.rmtree("temp_preprocessed")
            
            return jsonify({
                'success': True,
                'raw_text': best_text,
                'items': items,
                'total_items': len(items),
                'confidence': llm_results['confidence'],
                'ocr_confidence': best_confidence,
                'processing_info': {
                    'ocr_method': ocr_method,
                    'handwritten_ocr_available': handwritten_ocr.is_available(),
                    'preprocessing_used': ocr_method.startswith('traditional'),
                    'best_confidence': best_confidence
                }
            })
            
        except Exception as e:
            # Clean up temp files on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists("temp_preprocessed"):
                import shutil
                shutil.rmtree("temp_preprocessed")
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
        
        # Process with LLM (lazy load)
        llm_processor = get_llm_processor()
        llm_results = llm_processor.process_text(text)
        
        # Extract items
        items = []
        for item in llm_results['extracted_items']:
            items.append({
                'item_name': item['item_name'],
                'quantity': item['quantity'],
                'unit': item['unit'],
                'confidence': item['confidence']
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
        
        for item in items:
            item_name = item.get('item_name', '')
            quantity = item.get('quantity', '')
            unit = item.get('unit', '')
            
            if not item_name:
                continue
            
            # Search for the item
            search_query = f"{item_name} {quantity} {unit}".strip()
            
            item_results = {
                'item_name': item_name,
                'quantity': quantity,
                'unit': unit,
                'search_query': search_query,
                'platforms': {}
            }
            
            # Use the price scraper for smart matching
            from price_scraper import PriceScraper
            price_scraper = PriceScraper()
            
            # Create user item dictionary
            user_item = {
                'item_name': item_name,
                'quantity': quantity,
                'unit': unit
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
