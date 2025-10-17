#!/usr/bin/env python3
"""
Test script for Blinkit scraper
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path to allow direct imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the modules directly
from blinkit_scraper import BlinkitScraper
from utils import build_blinkit_url, build_blinkit_alternative_urls


def test_blinkit_scraper():
    """Test the Blinkit scraper functionality."""
    print("=" * 60)
    print("🧪 Testing Blinkit Scraper")
    print("=" * 60)
    
    # Test URL building
    print("📊 Testing URL building...")
    test_item = "curd"
    url = build_blinkit_url(test_item)
    print(f"Primary URL: {url}")
    
    # Show alternative URLs
    alt_urls = build_blinkit_alternative_urls(test_item)
    print(f"Alternative URLs to try: {len(alt_urls)}")
    for i, alt_url in enumerate(alt_urls[:3], 1):  # Show first 3
        print(f"  {i}. {alt_url}")
    if len(alt_urls) > 3:
        print(f"  ... and {len(alt_urls) - 3} more")
    
    # Test scraper initialization
    print("\n📊 Testing scraper initialization...")
    try:
        scraper = BlinkitScraper()
        print("✅ Scraper initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing scraper: {e}")
        return False
    
    # Test API data fetching (this might fail due to API restrictions)
    print("\n📊 Testing API data fetching...")
    try:
        # Test primary URL first
        print(f"Testing primary URL: {url}")
        api_data = scraper.fetch_api_data(url)
        
        if not api_data:
            print("⚠️  Primary URL failed, testing alternative URLs...")
            alt_urls = build_blinkit_alternative_urls(test_item)
            
            for i, alt_url in enumerate(alt_urls[:5], 1):  # Test first 5 alternatives
                print(f"Testing alternative {i}: {alt_url}")
                api_data = scraper.fetch_api_data(alt_url)
                if api_data:
                    print(f"✅ Success with alternative URL {i}")
                    break
                else:
                    print(f"❌ Alternative URL {i} failed")
        
        if api_data:
            print("✅ API data fetched successfully")
            print(f"Response keys: {list(api_data.keys())}")
            
            # Test parsing
            print("\n📊 Testing data parsing...")
            products = scraper.parse_results(api_data)
            print(f"✅ Parsed {len(products)} products")
            
            if products:
                print("\n📋 Sample product:")
                sample_product = products[0]
                for key, value in sample_product.items():
                    print(f"  {key}: {value}")
        else:
            print("⚠️  No API data received from any endpoint")
            print("This could be due to:")
            print("  - API restrictions or authentication requirements")
            print("  - Incorrect API endpoints")
            print("  - Network connectivity issues")
            print("  - Blinkit's anti-bot measures")
    except Exception as e:
        print(f"⚠️  API test failed: {e}")
        print("This is expected if the API endpoints are incorrect or restricted")
    
    # Clean up
    if 'scraper' in locals():
        scraper.session.close()
    
    print("\n" + "=" * 60)
    print("🏁 Test completed")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_blinkit_scraper()
