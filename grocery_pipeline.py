"""
Complete Grocery List Processing Pipeline
1. OCR: Handwritten grocery list → Text
2. Parser: Text → Items and quantities
3. Price Scraper: Items → Prices from Amazon/D-Mart
4. Optimizer: Create shopping list with best prices
"""
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

from ocr_processor import OCRProcessor
from price_scraper import PriceScraper

class GroceryPipeline:
    """Complete pipeline for processing handwritten grocery lists"""
    
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.price_scraper = PriceScraper()
        self.output_dir = "pipeline_results"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_handwritten_list(self, image_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: OCR → Parse → Price Scrape → Optimize
        
        Args:
            image_path: Path to handwritten grocery list image
            
        Returns:
            Complete processing results
        """
        print("=" * 60)
        print("GROCERY LIST PROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: OCR Processing
        print("\nStep 1: OCR Processing")
        print("-" * 30)
        grocery_dict = self.ocr_processor.process_grocery_list(image_path)
        
        if not grocery_dict:
            print("No items found in the grocery list")
            return {}
        
        print(f"Extracted {len(grocery_dict)} items:")
        for item, quantity in grocery_dict.items():
            print(f"  - {item}: {quantity}")
        
        # Step 2: Price Scraping
        print("\nStep 2: Price Scraping")
        print("-" * 30)
        grocery_prices = self.price_scraper.scrape_grocery_list_prices(grocery_dict)
        
        # Step 3: Shopping List Optimization
        print("\nStep 3: Shopping List Optimization")
        print("-" * 30)
        shopping_list = self.price_scraper.create_shopping_list(grocery_prices)
        
        # Step 4: Create Final Results
        print("\nStep 4: Creating Final Results")
        print("-" * 30)
        final_results = {
            'pipeline_info': {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'total_items': len(grocery_dict)
            },
            'extracted_items': grocery_dict,
            'price_analysis': grocery_prices,
            'shopping_list': shopping_list,
            'summary': self._create_summary(grocery_prices, shopping_list)
        }
        
        # Step 5: Save Results
        print("\nStep 5: Saving Results")
        print("-" * 30)
        self._save_all_results(final_results)
        
        return final_results
    
    def _create_summary(self, grocery_prices: Dict[str, Any], shopping_list: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the processing results"""
        summary = {
            'total_items': grocery_prices['total_items'],
            'items_with_prices': grocery_prices['summary']['items_with_prices'],
            'total_estimated_cost': grocery_prices['summary']['total_estimated_cost'],
            'platforms_available': len(grocery_prices['summary']['platforms_used']),
            'cheapest_platform': None,
            'cost_breakdown': {}
        }
        
        # Add platform cost breakdown
        if shopping_list['platforms']:
            summary['cost_breakdown'] = {
                platform: info['total_cost'] 
                for platform, info in shopping_list['platforms'].items()
            }
            
            # Find cheapest platform
            cheapest = min(shopping_list['platforms'].items(), 
                          key=lambda x: x[1]['total_cost'])
            summary['cheapest_platform'] = cheapest[0]
        
        return summary
    
    def _save_all_results(self, results: Dict[str, Any]):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        complete_file = os.path.join(self.output_dir, f"complete_results_{timestamp}.json")
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save extracted items
        items_file = os.path.join(self.output_dir, f"extracted_items_{timestamp}.json")
        with open(items_file, 'w', encoding='utf-8') as f:
            json.dump(results['extracted_items'], f, indent=2, ensure_ascii=False)
        
        # Save shopping list
        shopping_file = os.path.join(self.output_dir, f"shopping_list_{timestamp}.json")
        with open(shopping_file, 'w', encoding='utf-8') as f:
            json.dump(results['shopping_list'], f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f"summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results['summary'], f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to:")
        print(f"- {complete_file}")
        print(f"- {items_file}")
        print(f"- {shopping_file}")
        print(f"- {summary_file}")
    
    def display_results(self, results: Dict[str, Any]):
        """Display processing results in a user-friendly format"""
        print("\n" + "=" * 60)
        print("PROCESSING RESULTS")
        print("=" * 60)
        
        # Summary
        summary = results['summary']
        print(f"\nSUMMARY:")
        print(f"  Total Items: {summary['total_items']}")
        print(f"  Items with Prices: {summary['items_with_prices']}")
        print(f"  Total Estimated Cost: Rs. {summary['total_estimated_cost']:.2f}")
        print(f"  Cheapest Platform: {summary['cheapest_platform'] or 'N/A'}")
        
        # Extracted Items
        print(f"\nEXTRACTED ITEMS:")
        for item, quantity in results['extracted_items'].items():
            print(f"  - {item}: {quantity}")
        
        # Shopping List by Platform
        if results['shopping_list']['platforms']:
            print(f"\nSHOPPING LIST BY PLATFORM:")
            for platform, info in results['shopping_list']['platforms'].items():
                print(f"\n  {platform.upper()}:")
                print(f"    Total Cost: Rs. {info['total_cost']:.2f}")
                print(f"    Items: {info['item_count']}")
                for item_info in info['items']:
                    print(f"      - {item_info['item']} ({item_info['quantity']}) - Rs. {item_info['total_cost']:.2f}")
        
        # Recommendations
        if results['shopping_list']['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for rec in results['shopping_list']['recommendations']:
                print(f"  - {rec['type']}: {rec['platform']} - Rs. {rec['total_cost']:.2f}")

def main():
    """Main function for testing the pipeline"""
    pipeline = GroceryPipeline()
    
    # Test with sample image if available
    test_image = "sample_grocery_list.jpg"
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    if os.path.exists(test_image):
        print(f"Processing grocery list: {test_image}")
        results = pipeline.process_handwritten_list(test_image)
        
        if results:
            pipeline.display_results(results)
        else:
            print("No results generated")
    else:
        print(f"Image not found: {test_image}")
        print("\nUsage:")
        print("  python grocery_pipeline.py <image_path>")
        print("\nExample:")
        print("  python grocery_pipeline.py my_grocery_list.jpg")

if __name__ == "__main__":
    main()
