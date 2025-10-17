#!/usr/bin/env python3
"""
Flask Backend for Grocery OCR Pipeline
Integrates OCR model, scraper, and provides API endpoints
"""

from flask import Flask, request, jsonify, send_from_directory
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

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from grocery_ocr_llm_model import OCRProcessor, LLMProcessor, GroceryOCRModel
from Scrapper.amazon_scraper import AmazonScraper
from Scrapper.blinkit_scraper import BlinkitScraper
from Scrapper.utils import build_amazon_url, build_blinkit_url
from specialized_preprocessing import GroceryListPreprocessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model components
ocr_processor = None
llm_processor = None
model = None
amazon_scraper = None
blinkit_scraper = None
preprocessor = None

def initialize_components():
    """Initialize OCR model and scraper components"""
    global ocr_processor, llm_processor, model, amazon_scraper, blinkit_scraper, preprocessor
    
    try:
        logger.info("Initializing OCR components...")
        ocr_processor = OCRProcessor()
        llm_processor = LLMProcessor()
        model = GroceryOCRModel(ocr_processor, llm_processor)
        
        # Initialize specialized preprocessor
        preprocessor = GroceryListPreprocessor()
        
        # Load trained model if available
        model_path = "grocery_ocr_model.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Loaded trained model")
        else:
            logger.warning("⚠️ No trained model found, using untrained model")
        
        model.eval()
        
        # Initialize scrapers
        amazon_scraper = AmazonScraper()
        blinkit_scraper = BlinkitScraper()
        
        logger.info("✅ All components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Error initializing components: {e}")
        raise

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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'ocr_processor': ocr_processor is not None,
            'llm_processor': llm_processor is not None,
            'model': model is not None,
            'amazon_scraper': amazon_scraper is not None,
            'blinkit_scraper': blinkit_scraper is not None,
            'preprocessor': preprocessor is not None
        }
    })

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
            # Use specialized preprocessing for better OCR results
            processed_versions = preprocessor.preprocess_grocery_list(temp_path, "temp_preprocessed")
            
            # Try OCR on multiple preprocessed versions
            best_text = ""
            best_confidence = 0
            best_results = []
            
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
                        # Process with OCR directly on the preprocessed file
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
                                best_results = ocr_results
                                
                                logger.info(f"Better OCR result from {version_name}: confidence={avg_confidence:.3f}, text_length={len(predicted_text)}")
                
                except Exception as e:
                    logger.warning(f"Error processing {version_name}: {e}")
                    continue
            
            # If no good results from preprocessed versions, fall back to original
            if not best_text.strip():
                logger.info("Falling back to original image processing")
                processed_image = ocr_processor.preprocess_image_optimal(temp_path)
                ocr_results = ocr_processor.extract_text_optimal(processed_image)
                best_text = ' '.join(text for text, _, _ in ocr_results)
                best_results = ocr_results
                best_confidence = sum(conf for _, conf, _ in ocr_results) / len(ocr_results) if ocr_results else 0
            
            # Process with LLM
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
                    'preprocessing_used': True,
                    'versions_tested': len(test_versions),
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
        
        # Process with LLM
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
        platforms = data.get('platforms', ['amazon', 'blinkit', 'zepto', 'bigbasket'])  # Default to all platforms
        
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
            
            # Process results for each platform
            for platform in platforms:
                if platform in best_prices['platforms']:
                    platform_info = best_prices['platforms'][platform]
                    platform_products = [{
                        'name': platform_info.get('product_name', ''),
                        'price': platform_info.get('price_text', ''),
                        'unit_price': platform_info.get('best_price', 0),
                        'calculated_total': platform_info.get('calculated_total', 0),
                        'rating': platform_info.get('rating', ''),
                        'review_count': platform_info.get('review_count', ''),
                        'product_url': platform_info.get('product_url', ''),
                        'image_url': platform_info.get('image_url', ''),
                        'brand': platform_info.get('brand', ''),
                        'variant': platform_info.get('variant', ''),
                        'inventory': platform_info.get('inventory', 0),
                        'eta': platform_info.get('eta', ''),
                        'match_score': platform_info.get('match_score', 0),
                        'total_products_found': platform_info.get('total_products', 0)
                    }]
                    item_results['platforms'][platform] = {
                        'products': platform_products
                    }
                else:
                    item_results['platforms'][platform] = {
                        'error': 'No products found',
                        'products': []
                    }
            
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
        'confidence_scoring': 'Uses average confidence across all detected text regions'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
