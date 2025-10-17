"""
OCR Processor for Handwritten Grocery Lists
Converts handwritten grocery lists to structured data
"""
import cv2
import numpy as np
import re
import json
from typing import Dict, List, Tuple
import os

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. Install with: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract not available. Install with: pip install pytesseract")

class OCRProcessor:
    """Process handwritten grocery lists using OCR"""
    
    def __init__(self):
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(['en'])
                print("EasyOCR initialized successfully")
            except Exception as e:
                print(f"EasyOCR initialization failed: {e}")
                self.reader = None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_easyocr(self, image_path: str) -> List[Tuple[str, float]]:
        """Extract text using EasyOCR"""
        if not self.reader:
            return []
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Extract text
            results = self.reader.readtext(processed_img)
            
            # Extract text and confidence
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    extracted_text.append((text.strip(), confidence))
            
            return extracted_text
        except Exception as e:
            print(f"EasyOCR extraction failed: {e}")
            return []
    
    def extract_text_tesseract(self, image_path: str) -> str:
        """Extract text using Tesseract"""
        if not TESSERACT_AVAILABLE:
            return ""
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Extract text
            text = pytesseract.image_to_string(processed_img, config='--psm 6')
            return text
        except Exception as e:
            print(f"Tesseract extraction failed: {e}")
            return ""
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using available OCR methods"""
        print(f"Processing image: {image_path}")
        
        # Try EasyOCR first
        if self.reader:
            print("Using EasyOCR...")
            easyocr_results = self.extract_text_easyocr(image_path)
            if easyocr_results:
                # Combine all text
                text = " ".join([result[0] for result in easyocr_results])
                print(f"EasyOCR extracted: {text}")
                return text
        
        # Fallback to Tesseract
        if TESSERACT_AVAILABLE:
            print("Using Tesseract...")
            text = self.extract_text_tesseract(image_path)
            if text.strip():
                print(f"Tesseract extracted: {text}")
                return text
        
        print("No OCR method available or extraction failed")
        return ""
    
    def parse_grocery_list(self, text: str) -> Dict[str, int]:
        """
        Parse grocery list text to extract items and quantities
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Dictionary with items as keys and quantities as values
        """
        print(f"Parsing grocery list text: {text}")
        
        # Clean the text
        text = text.lower().strip()
        
        # Common grocery items and their variations
        grocery_patterns = {
            'milk': ['milk', 'doodh', 'dudh'],
            'bread': ['bread', 'roti', 'chapati'],
            'butter': ['butter', 'makhan', 'ghee'],
            'oil': ['oil', 'tel', 'cooking oil', 'vegetable oil'],
            'rice': ['rice', 'chawal', 'basmati'],
            'wheat': ['wheat', 'gehun', 'atta', 'flour'],
            'sugar': ['sugar', 'chini', 'gur'],
            'salt': ['salt', 'namak'],
            'onion': ['onion', 'pyaaz', 'pyaz'],
            'tomato': ['tomato', 'tamatar'],
            'potato': ['potato', 'aloo', 'batata'],
            'eggs': ['eggs', 'anda', 'ande'],
            'chicken': ['chicken', 'murga', 'murgi'],
            'fish': ['fish', 'machli'],
            'banana': ['banana', 'kela'],
            'apple': ['apple', 'seb'],
            'orange': ['orange', 'santra'],
            'lemon': ['lemon', 'nimbu'],
            'ginger': ['ginger', 'adrak'],
            'garlic': ['garlic', 'lehsun'],
            'curd': ['curd', 'dahi', 'yogurt'],
            'cheese': ['cheese', 'paneer'],
            'tea': ['tea', 'chai'],
            'coffee': ['coffee', 'kapi'],
            'biscuits': ['biscuits', 'biscuit', 'cookies'],
            'chips': ['chips', 'namkeen'],
            'soap': ['soap', 'sabun'],
            'shampoo': ['shampoo', 'shampo'],
            'toothpaste': ['toothpaste', 'tooth brush', 'brush']
        }
        
        # Quantity patterns
        quantity_patterns = [
            r'(\d+)\s*(kg|kilo|kilogram)',
            r'(\d+)\s*(g|gram|gm)',
            r'(\d+)\s*(l|liter|litre)',
            r'(\d+)\s*(ml|milliliter)',
            r'(\d+)\s*(pack|packs|pkt|packet)',
            r'(\d+)\s*(dozen|dz)',
            r'(\d+)\s*(pcs|pieces|piece)',
            r'(\d+)\s*(box|boxes)',
            r'(\d+)\s*(bottle|bottles)',
            r'(\d+)\s*(tin|tins|can|cans)',
            r'(\d+)\s*(bag|bags)',
            r'(\d+)\s*(kg|g|l|ml|pack|dozen|pcs|box|bottle|tin|bag)',
            r'(\d+)'  # Just numbers
        ]
        
        # Parse the text
        grocery_dict = {}
        
        # Split text into lines
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Find quantities
            quantity = 1  # Default quantity
            for pattern in quantity_patterns:
                match = re.search(pattern, line)
                if match:
                    quantity = int(match.group(1))
                    # Remove quantity from line for item matching
                    line = re.sub(pattern, '', line).strip()
                    break
            
            # Find grocery items
            for item, variations in grocery_patterns.items():
                for variation in variations:
                    if variation in line:
                        grocery_dict[item] = quantity
                        print(f"Found: {item} - {quantity}")
                        break
                if item in grocery_dict:
                    break
        
        # If no items found, try to extract any words that might be items
        if not grocery_dict:
            print("No standard items found, trying to extract any words...")
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            for word in words:
                if len(word) > 2:  # Skip very short words
                    grocery_dict[word] = 1
        
        return grocery_dict
    
    def process_grocery_list(self, image_path: str) -> Dict[str, int]:
        """
        Complete pipeline: OCR + parsing
        
        Args:
            image_path: Path to handwritten grocery list image
            
        Returns:
            Dictionary with items and quantities
        """
        print(f"Processing grocery list: {image_path}")
        
        # Extract text using OCR
        text = self.extract_text(image_path)
        
        if not text.strip():
            print("No text extracted from image")
            return {}
        
        # Parse the text to get grocery items
        grocery_dict = self.parse_grocery_list(text)
        
        return grocery_dict
    
    def save_grocery_list(self, grocery_dict: Dict[str, int], output_path: str):
        """Save grocery list to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(grocery_dict, f, indent=2, ensure_ascii=False)
        print(f"Grocery list saved to: {output_path}")

def main():
    """Test the OCR processor"""
    processor = OCRProcessor()
    
    # Test with sample image if available
    test_image = "sample_grocery_list.jpg"
    if os.path.exists(test_image):
        grocery_dict = processor.process_grocery_list(test_image)
        print(f"Extracted grocery list: {grocery_dict}")
        
        # Save results
        processor.save_grocery_list(grocery_dict, "extracted_grocery_list.json")
    else:
        print(f"Test image {test_image} not found")
        print("Please provide a handwritten grocery list image to test")

if __name__ == "__main__":
    main()
