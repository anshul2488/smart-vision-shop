#!/usr/bin/env python3
"""
Specialized image preprocessing for handwritten grocery lists
Optimized for lined paper with blue lines and red margin
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from typing import List

class GroceryListPreprocessor:
    def __init__(self):
        self.line_color_lower = np.array([100, 50, 50])  # Blue lines
        self.line_color_upper = np.array([130, 255, 255])
        self.margin_color_lower = np.array([0, 50, 50])   # Red margin
        self.margin_color_upper = np.array([10, 255, 255])
    
    def remove_lines(self, image: np.ndarray) -> np.ndarray:
        """Remove blue lines and red margin from lined paper"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue lines
        blue_mask = cv2.inRange(hsv, self.line_color_lower, self.line_color_upper)
        
        # Create mask for red margin
        red_mask = cv2.inRange(hsv, self.margin_color_lower, self.margin_color_upper)
        
        # Combine masks
        line_mask = cv2.bitwise_or(blue_mask, red_mask)
        
        # Invert mask (we want to keep everything except lines)
        mask_inv = cv2.bitwise_not(line_mask)
        
        # Apply mask to remove lines
        result = cv2.bitwise_and(image, image, mask=mask_inv)
        
        # Fill removed areas with white
        result[line_mask > 0] = [255, 255, 255]
        
        return result
    
    def enhance_handwriting(self, image: np.ndarray) -> np.ndarray:
        """Enhance handwritten text specifically"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(filtered)
        
        # Apply gamma correction
        gamma = 1.5
        gamma_corrected = np.power(enhanced / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(gamma_corrected)
        
        return gamma_corrected
    
    def adaptive_threshold_handwriting(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding optimized for handwriting"""
        # Multiple thresholding approaches
        # Method 1: Gaussian adaptive threshold
        gaussian = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Method 2: Mean adaptive threshold
        mean = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 15, 3
        )
        
        # Method 3: Otsu threshold
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine methods
        combined = cv2.bitwise_and(gaussian, mean)
        combined = cv2.bitwise_and(combined, otsu)
        
        return combined
    
    def remove_noise_handwriting(self, image: np.ndarray) -> np.ndarray:
        """Remove noise specific to handwritten text"""
        # Morphological operations
        kernel_noise = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_noise)
        
        # Fill small holes
        kernel_fill = np.ones((3, 3), np.uint8)
        filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_fill)
        
        # Remove very small connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled, connectivity=8)
        
        # Create mask for components larger than threshold
        mask = np.zeros_like(filled)
        min_size = 30  # Minimum component size
        
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == i] = 255
        
        return mask
    
    def deskew_handwriting(self, image: np.ndarray) -> np.ndarray:
        """Deskew image based on text lines"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Get all contour points
        all_points = np.vstack(contours)
        
        # Fit a line to the points
        if len(all_points) > 1:
            # Use PCA to find the main direction
            mean = np.mean(all_points, axis=0)
            centered = all_points - mean
            
            # Calculate covariance matrix
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # Get the angle of the principal component
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            angle_degrees = np.degrees(angle)
            
            # Only rotate if angle is significant
            if abs(angle_degrees) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                       flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
                return rotated
        
        return image
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Final enhancement for OCR"""
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        # Convert back to numpy
        enhanced = np.array(pil_image)
        
        return enhanced
    
    def preprocess_grocery_list(self, image_path: str, output_dir: str = "preprocessed") -> List[tuple]:
        """Complete preprocessing pipeline for grocery lists"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Original shape: {image.shape}")
        
        # Step 1: Remove lines
        no_lines = self.remove_lines(image)
        cv2.imwrite(os.path.join(output_dir, "01_no_lines.jpg"), no_lines)
        
        # Step 2: Enhance handwriting
        enhanced = self.enhance_handwriting(no_lines)
        cv2.imwrite(os.path.join(output_dir, "02_enhanced.jpg"), enhanced)
        
        # Step 3: Adaptive threshold
        thresholded = self.adaptive_threshold_handwriting(enhanced)
        cv2.imwrite(os.path.join(output_dir, "03_thresholded.jpg"), thresholded)
        
        # Step 4: Remove noise
        denoised = self.remove_noise_handwriting(thresholded)
        cv2.imwrite(os.path.join(output_dir, "04_denoised.jpg"), denoised)
        
        # Step 5: Deskew
        deskewed = self.deskew_handwriting(denoised)
        cv2.imwrite(os.path.join(output_dir, "05_deskewed.jpg"), deskewed)
        
        # Step 6: Final enhancement
        final = self.enhance_for_ocr(deskewed)
        cv2.imwrite(os.path.join(output_dir, "06_final.jpg"), final)
        
        # Create inverted version
        inverted = 255 - final
        cv2.imwrite(os.path.join(output_dir, "07_inverted.jpg"), inverted)
        
        # Create high contrast version
        high_contrast = cv2.convertScaleAbs(final, alpha=2.0, beta=0)
        cv2.imwrite(os.path.join(output_dir, "08_high_contrast.jpg"), high_contrast)
        
        # Ensure all images are 2D grayscale for OCR
        def ensure_grayscale(img):
            if len(img.shape) == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        
        # Return all processed versions (all as 2D grayscale)
        processed_versions = [
            ("Original", ensure_grayscale(image)),
            ("No_Lines", ensure_grayscale(no_lines)),
            ("Enhanced", enhanced),  # Already grayscale
            ("Thresholded", thresholded),  # Already grayscale
            ("Denoised", denoised),  # Already grayscale
            ("Deskewed", deskewed),  # Already grayscale
            ("Final", final),  # Already grayscale
            ("Inverted", inverted),  # Already grayscale
            ("High_Contrast", high_contrast)  # Already grayscale
        ]
        
        print(f"✅ Preprocessing completed. {len(processed_versions)} versions created in '{output_dir}' folder")
        
        return processed_versions

def main():
    """Main function to preprocess grocery list image"""
    print("Specialized Grocery List Preprocessing")
    print("=" * 40)
    
    processor = GroceryListPreprocessor()
    
    # Get image path
    image_path = input("Enter path to your grocery list image: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        # Preprocess the image
        processed_versions = processor.preprocess_grocery_list(image_path)
        
        print(f"\n📁 Preprocessed images saved in 'preprocessed' folder:")
        for i, (name, _) in enumerate(processed_versions, 1):
            print(f"  {i}. {name}")
        
        print(f"\n💡 Try OCR on these versions:")
        print(f"  - 06_final.jpg (recommended)")
        print(f"  - 07_inverted.jpg (if text is dark on light)")
        print(f"  - 08_high_contrast.jpg (for low contrast)")
        
        print(f"\n🔧 Next steps:")
        print(f"  1. Run: python alternative_ocr_methods.py")
        print(f"  2. Use one of the preprocessed images")
        print(f"  3. Compare results across different versions")
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")

if __name__ == "__main__":
    main()
