#!/usr/bin/env python3
"""
Comprehensive OCR testing for handwritten grocery lists
Tests multiple engines, preprocessing methods, and configurations
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class TestResult:
    method: str
    preprocessing: str
    engine: str
    text_detected: str
    confidence: float
    num_detections: int
    processing_time: float

class ComprehensiveOCRTester:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.results = []
        
        # Initialize OCR engines
        self.engines = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all available OCR engines"""
        # PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.engines['paddleocr_standard'] = PaddleOCR(
                use_angle_cls=True, lang='en', use_gpu=False, show_log=False
            )
            self.engines['paddleocr_handwritten'] = PaddleOCR(
                use_angle_cls=True, lang='en', use_gpu=False, show_log=False,
                det_limit_side_len=1280, drop_score=0.1, max_text_length=50
            )
            self.logger.info("✅ PaddleOCR engines initialized")
        except Exception as e:
            self.logger.warning(f"PaddleOCR not available: {e}")
        
        # EasyOCR
        try:
            import easyocr
            self.engines['easyocr_standard'] = easyocr.Reader(['en'], gpu=False)
            self.engines['easyocr_sensitive'] = easyocr.Reader(['en'], gpu=False)
            self.logger.info("✅ EasyOCR engines initialized")
        except Exception as e:
            self.logger.warning(f"EasyOCR not available: {e}")
        
        # Tesseract
        try:
            import pytesseract
            self.engines['tesseract'] = pytesseract
            self.logger.info("✅ Tesseract initialized")
        except Exception as e:
            self.logger.warning(f"Tesseract not available: {e}")
    
    def get_preprocessing_methods(self, image_path: str) -> List[Tuple[str, np.ndarray]]:
        """Get all preprocessing methods"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        methods = []
        
        # Original
        methods.append(("Original", image))
        
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        methods.append(("Grayscale", gray))
        
        # Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        methods.append(("Enhanced", enhanced))
        
        # Inverted
        inverted = 255 - enhanced
        methods.append(("Inverted", inverted))
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        methods.append(("Adaptive", adaptive))
        
        # Otsu threshold
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("Otsu", otsu))
        
        # Morphological operations
        kernel = np.ones((2,2), np.uint8)
        morphed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        methods.append(("Morphed", morphed))
        
        # Gaussian blur + threshold
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, gaussian_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("Gaussian", gaussian_thresh))
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        methods.append(("Edges", edges))
        
        # High contrast
        high_contrast = cv2.convertScaleAbs(enhanced, alpha=2.0, beta=0)
        methods.append(("High_Contrast", high_contrast))
        
        return methods
    
    def test_paddleocr(self, image: np.ndarray, engine_name: str, preprocessing_name: str) -> TestResult:
        """Test PaddleOCR"""
        import time
        start_time = time.time()
        
        try:
            ocr = self.engines[engine_name]
            detections = ocr.ocr(image, cls=True)
            
            text_parts = []
            confidences = []
            
            if detections and detections[0]:
                for detection in detections[0]:
                    text, confidence = detection[1]
                    text_parts.append(text.strip())
                    confidences.append(confidence)
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            processing_time = time.time() - start_time
            
            return TestResult(
                method=f"PaddleOCR-{engine_name.split('_')[1]}",
                preprocessing=preprocessing_name,
                engine="PaddleOCR",
                text_detected=full_text,
                confidence=avg_confidence,
                num_detections=len(text_parts),
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.warning(f"PaddleOCR {engine_name} failed: {e}")
            return TestResult(
                method=f"PaddleOCR-{engine_name.split('_')[1]}",
                preprocessing=preprocessing_name,
                engine="PaddleOCR",
                text_detected="",
                confidence=0.0,
                num_detections=0,
                processing_time=time.time() - start_time
            )
    
    def test_easyocr(self, image: np.ndarray, engine_name: str, preprocessing_name: str) -> TestResult:
        """Test EasyOCR"""
        import time
        start_time = time.time()
        
        try:
            reader = self.engines[engine_name]
            
            # Try different parameter sets
            param_sets = [
                {'width_ths': 0.1, 'height_ths': 0.1, 'paragraph': False},
                {'width_ths': 0.3, 'height_ths': 0.3, 'paragraph': False},
                {'width_ths': 0.5, 'height_ths': 0.5, 'paragraph': True},
            ]
            
            best_text = ""
            best_confidence = 0
            best_detections = 0
            
            for params in param_sets:
                try:
                    detections = reader.readtext(image, **params)
                    
                    text_parts = []
                    confidences = []
                    
                    for detection in detections:
                        bbox, text, confidence = detection
                        text_parts.append(text.strip())
                        confidences.append(confidence)
                    
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        if avg_conf > best_confidence:
                            best_confidence = avg_conf
                            best_text = ' '.join(text_parts)
                            best_detections = len(text_parts)
                            
                except Exception as e:
                    continue
            
            processing_time = time.time() - start_time
            
            return TestResult(
                method=f"EasyOCR-{engine_name.split('_')[1]}",
                preprocessing=preprocessing_name,
                engine="EasyOCR",
                text_detected=best_text,
                confidence=best_confidence,
                num_detections=best_detections,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.warning(f"EasyOCR {engine_name} failed: {e}")
            return TestResult(
                method=f"EasyOCR-{engine_name.split('_')[1]}",
                preprocessing=preprocessing_name,
                engine="EasyOCR",
                text_detected="",
                confidence=0.0,
                num_detections=0,
                processing_time=time.time() - start_time
            )
    
    def test_tesseract(self, image: np.ndarray, engine_name: str, preprocessing_name: str) -> TestResult:
        """Test Tesseract"""
        import time
        start_time = time.time()
        
        try:
            tesseract = self.engines[engine_name]
            
            # Try different OCR modes
            configs = [
                '--oem 3 --psm 6',  # Uniform block of text
                '--oem 3 --psm 8',  # Single word
                '--oem 3 --psm 13', # Raw line
                '--oem 3 --psm 11', # Sparse text
            ]
            
            best_text = ""
            best_confidence = 0
            best_detections = 0
            
            for config in configs:
                try:
                    text = tesseract.image_to_string(image, config=config)
                    if text.strip():
                        # Get confidence data
                        data = tesseract.image_to_data(image, config=config, output_type=tesseract.Output.DICT)
                        
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        if confidences:
                            avg_conf = sum(confidences) / len(confidences) / 100.0
                            if avg_conf > best_confidence:
                                best_confidence = avg_conf
                                best_text = text.strip()
                                best_detections = len([t for t in data['text'] if t.strip()])
                                
                except Exception as e:
                    continue
            
            processing_time = time.time() - start_time
            
            return TestResult(
                method="Tesseract",
                preprocessing=preprocessing_name,
                engine="Tesseract",
                text_detected=best_text,
                confidence=best_confidence,
                num_detections=best_detections,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.warning(f"Tesseract failed: {e}")
            return TestResult(
                method="Tesseract",
                preprocessing=preprocessing_name,
                engine="Tesseract",
                text_detected="",
                confidence=0.0,
                num_detections=0,
                processing_time=time.time() - start_time
            )
    
    def run_comprehensive_test(self, image_path: str) -> List[TestResult]:
        """Run comprehensive OCR test"""
        self.logger.info(f"Starting comprehensive OCR test on: {image_path}")
        
        # Get all preprocessing methods
        preprocessing_methods = self.get_preprocessing_methods(image_path)
        
        results = []
        
        for preprocessing_name, processed_image in preprocessing_methods:
            self.logger.info(f"Testing preprocessing: {preprocessing_name}")
            
            # Test each engine
            for engine_name in self.engines.keys():
                if engine_name.startswith('paddleocr'):
                    result = self.test_paddleocr(processed_image, engine_name, preprocessing_name)
                elif engine_name.startswith('easyocr'):
                    result = self.test_easyocr(processed_image, engine_name, preprocessing_name)
                elif engine_name == 'tesseract':
                    result = self.test_tesseract(processed_image, engine_name, preprocessing_name)
                else:
                    continue
                
                results.append(result)
                self.logger.info(f"  {result.method} on {preprocessing_name}: "
                               f"{result.num_detections} detections, "
                               f"confidence: {result.confidence:.3f}")
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict:
        """Analyze test results and find best combinations"""
        if not self.results:
            return {}
        
        # Sort by confidence * num_detections (composite score)
        scored_results = []
        for result in self.results:
            if result.num_detections > 0:
                score = result.confidence * result.num_detections
                scored_results.append((score, result))
        
        scored_results.sort(reverse=True, key=lambda x: x[0])
        
        # Get top 10 results
        top_results = [result for score, result in scored_results[:10]]
        
        # Analyze by engine
        engine_stats = {}
        for result in self.results:
            if result.engine not in engine_stats:
                engine_stats[result.engine] = []
            engine_stats[result.engine].append(result)
        
        # Analyze by preprocessing
        preprocessing_stats = {}
        for result in self.results:
            if result.preprocessing not in preprocessing_stats:
                preprocessing_stats[result.preprocessing] = []
            preprocessing_stats[result.preprocessing].append(result)
        
        analysis = {
            'top_results': [asdict(r) for r in top_results],
            'engine_stats': {
                engine: {
                    'count': len(results),
                    'avg_confidence': sum(r.confidence for r in results) / len(results),
                    'avg_detections': sum(r.num_detections for r in results) / len(results),
                    'best_result': max(results, key=lambda r: r.confidence * r.num_detections)
                }
                for engine, results in engine_stats.items()
            },
            'preprocessing_stats': {
                prep: {
                    'count': len(results),
                    'avg_confidence': sum(r.confidence for r in results) / len(results),
                    'avg_detections': sum(r.num_detections for r in results) / len(results),
                    'best_result': max(results, key=lambda r: r.confidence * r.num_detections)
                }
                for prep, results in preprocessing_stats.items()
            }
        }
        
        return analysis
    
    def save_results(self, output_file: str = "ocr_test_results.json"):
        """Save test results to JSON file"""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(r) for r in self.results],
            'analysis': self.analyze_results()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_file}")
    
    def print_summary(self):
        """Print test summary"""
        if not self.results:
            print("No results to display")
            return
        
        analysis = self.analyze_results()
        
        print("\n" + "="*60)
        print("COMPREHENSIVE OCR TEST RESULTS")
        print("="*60)
        
        print(f"\n📊 Top 5 Results:")
        print("-" * 40)
        for i, result_dict in enumerate(analysis['top_results'][:5], 1):
            result = TestResult(**result_dict)
            print(f"{i}. {result.method} + {result.preprocessing}")
            print(f"   Detections: {result.num_detections}, "
                  f"Confidence: {result.confidence:.3f}, "
                  f"Time: {result.processing_time:.2f}s")
            print(f"   Text: {result.text_detected[:100]}...")
            print()
        
        print(f"\n🔧 Best Engine: {max(analysis['engine_stats'].items(), key=lambda x: x[1]['avg_confidence'])[0]}")
        print(f"🎯 Best Preprocessing: {max(analysis['preprocessing_stats'].items(), key=lambda x: x[1]['avg_confidence'])[0]}")
        
        print(f"\n📈 Engine Performance:")
        print("-" * 30)
        for engine, stats in analysis['engine_stats'].items():
            print(f"{engine:15s}: {stats['avg_confidence']:.3f} avg confidence, "
                  f"{stats['avg_detections']:.1f} avg detections")

def main():
    """Main function"""
    print("Comprehensive OCR Testing for Handwritten Grocery Lists")
    print("=" * 60)
    
    tester = ComprehensiveOCRTester()
    
    # Get image path
    image_path = input("Enter path to your grocery list image: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test(image_path)
        
        # Print summary
        tester.print_summary()
        
        # Save results
        tester.save_results()
        
        print(f"\n✅ Test completed! {len(results)} combinations tested.")
        print(f"📁 Detailed results saved to: ocr_test_results.json")
        
    except Exception as e:
        print(f"❌ Error running test: {e}")

if __name__ == "__main__":
    main()
