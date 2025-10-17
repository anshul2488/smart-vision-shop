"""
Test script for Zepto integration
Tests the complete pipeline with Zepto scraper
"""
import sys
import os
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from price_scraper import PriceScraper

def test_zepto_integration():
    """Test the complete Zepto integration"""
    print("=" * 60)
    print("TESTING ZEPTO INTEGRATION")
    print("=" * 60)
    
    # Initialize price scraper
    scraper = PriceScraper()
    
    # Test items
    test_items = [
        "milk",
        "bread", 
        "eggs",
        "rice",
        "tomatoes"
    ]
    
    print(f"\nTesting with {len(test_items)} items:")
    for item in test_items:
        print(f"  - {item}")
    
    # Test individual item scraping
    print(f"\n{'='*40}")
    print("TESTING INDIVIDUAL ITEM SCRAPING")
    print(f"{'='*40}")
    
    for item in test_items[:2]:  # Test first 2 items
        print(f"\nScraping: {item}")
        results = scraper.scrape_item_prices(item, max_results=3)
        
        for platform, products in results.items():
            print(f"  {platform.upper()}: {len(products)} products")
            for i, product in enumerate(products[:2], 1):
                print(f"    {i}. {product.get('name', 'N/A')} - ₹{product.get('price', 'N/A')}")
    
    # Test grocery list scraping
    print(f"\n{'='*40}")
    print("TESTING GROCERY LIST SCRAPING")
    print(f"{'='*40}")
    
    grocery_dict = {
        "milk": 1,
        "bread": 2,
        "eggs": 12
    }
    
    print(f"Grocery list: {grocery_dict}")
    
    try:
        grocery_prices = scraper.scrape_grocery_list_prices(grocery_dict)
        
        if grocery_prices:
            print(f"\n✅ Successfully scraped prices for {len(grocery_prices)} items")
            
            # Show summary
            for item_name, item_data in grocery_prices.items():
                print(f"\n{item_name}:")
                if 'platforms' in item_data:
                    for platform, platform_data in item_data['platforms'].items():
                        if platform_data.get('calculated_total', 0) > 0:
                            print(f"  {platform}: ₹{platform_data['calculated_total']:.2f}")
                        else:
                            print(f"  {platform}: No products found")
                else:
                    print("  No platform data found")
        else:
            print("❌ No grocery prices found")
            
    except Exception as e:
        print(f"❌ Error in grocery list scraping: {str(e)}")
    
    # Test shopping list creation
    print(f"\n{'='*40}")
    print("TESTING SHOPPING LIST CREATION")
    print(f"{'='*40}")
    
    try:
        if grocery_prices:
            shopping_list = scraper.create_shopping_list(grocery_prices)
            
            if shopping_list and 'platforms' in shopping_list:
                print("✅ Shopping list created successfully")
                
                for platform, platform_data in shopping_list['platforms'].items():
                    total_cost = platform_data.get('total_cost', 0)
                    item_count = platform_data.get('item_count', 0)
                    print(f"  {platform}: ₹{total_cost:.2f} ({item_count} items)")
            else:
                print("❌ Shopping list creation failed")
        else:
            print("❌ Cannot create shopping list - no grocery prices")
            
    except Exception as e:
        print(f"❌ Error creating shopping list: {str(e)}")
    
    print(f"\n{'='*60}")
    print("ZEPTO INTEGRATION TEST COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_zepto_integration()
