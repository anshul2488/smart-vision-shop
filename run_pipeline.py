#!/usr/bin/env python3
"""
Complete Grocery OCR Pipeline Runner
Starts the Flask backend with all components
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import torch
        import cv2
        import easyocr
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements_flask.txt")
        return False

def check_model_file():
    """Check if the trained model file exists"""
    model_path = "grocery_ocr_model.pth"
    if os.path.exists(model_path):
        print(f"✅ Found trained model: {model_path}")
        return True
    else:
        print(f"⚠️ No trained model found at: {model_path}")
        print("The system will use an untrained model (less accurate)")
        return False

def start_backend():
    """Start the Flask backend"""
    print("🚀 Starting Flask backend...")
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def main():
    """Main function"""
    print("🛒 Grocery OCR Pipeline")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model file
    check_model_file()
    
    print("\n📋 Pipeline Components:")
    print("  - OCR Model: grocery_ocr_llm_model.py")
    print("  - Scraper: Scrapper/amazon_scraper.py")
    print("  - Frontend: Frontend/")
    print("  - Backend: app.py")
    
    print("\n🌐 Starting server...")
    print("  - Backend: http://localhost:5000")
    print("  - Frontend: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    
    # Wait a moment then open browser
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the backend
    start_backend()

if __name__ == "__main__":
    main()
