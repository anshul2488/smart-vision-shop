#!/usr/bin/env python3
"""
Amazon India Scraper - Main Entry Point

This script scrapes Amazon India product listings using Crawl4AI.
Usage: python main.py <item_name>
Example: python main.py oil
Example: python main.py "body lotion"
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scraper.amazon_scraper import AmazonScraper
from scraper.blinkit_scraper import BlinkitScraper
from scraper.utils import build_amazon_url, build_blinkit_url


def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("🛒 E-commerce Scraper")
    print("=" * 60)
    print("Scraping product listings from Amazon India and Blinkit")
    print("=" * 60)


def print_progress(message: str):
    """Print progress message with formatting."""
    print(f"📊 {message}")


def print_success(message: str):
    """Print success message with formatting."""
    print(f"✅ {message}")


def print_error(message: str):
    """Print error message with formatting."""
    print(f"❌ {message}")


def print_warning(message: str):
    """Print warning message with formatting."""
    print(f"⚠️  {message}")


def main():
    """Main function to orchestrate the scraping workflow."""
    print_banner()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Scrape product listings from Amazon India and Blinkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py oil --platform amazon
  python main.py "body lotion" --platform blinkit
  python main.py "mobile phone" --platform amazon
  python main.py oil  # Defaults to Amazon
        """
    )
    
    parser.add_argument(
        'item_name',
        help='Name of the item to search for'
    )
    
    parser.add_argument(
        '--platform',
        choices=['amazon', 'blinkit'],
        default='amazon',
        help='Platform to scrape (default: amazon)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.item_name.strip():
        print_error("Item name cannot be empty!")
        sys.exit(1)
    
    item_name = args.item_name.strip()
    output_dir = args.output_dir
    platform = args.platform.lower()
    
    try:
        # Show what we're searching for
        print_progress(f"Searching for: '{item_name}' on {platform.title()}")
        
        # Initialize the appropriate scraper
        if platform == 'blinkit':
            # Build and display the Blinkit URL
            search_url = build_blinkit_url(item_name)
            print_progress(f"Blinkit API URL: {search_url}")
            
            print_progress("Initializing Blinkit scraper...")
            scraper = BlinkitScraper()
            
            # Start scraping
            print_progress("Starting to scrape Blinkit...")
            print_progress("This may take a few moments...")
            
            # Scrape the products
            products = scraper.scrape_blinkit(item_name, output_dir)
        else:
            # Build and display the Amazon URL
            search_url = build_amazon_url(item_name)
            print_progress(f"Amazon URL: {search_url}")
            
            print_progress("Initializing Amazon scraper...")
            scraper = AmazonScraper()
            
            # Start scraping
            print_progress("Starting to scrape Amazon India...")
            print_progress("This may take a few moments...")
            
            # Scrape the products
            products = scraper.scrape_amazon(item_name, output_dir)
        
        # Display results
        if products:
            print_success(f"Scraping completed successfully!")
            print_success(f"Found {len(products)} products")
            
            # Show sample of results
            print("\n📋 Sample Results:")
            print("-" * 40)
            for i, product in enumerate(products[:3], 1):
                print(f"{i}. {product.get('name', 'N/A')[:50]}...")
                print(f"   Price: {product.get('price', 'N/A')}")
                print(f"   Rating: {product.get('rating', 'N/A')}")
                print(f"   Reviews: {product.get('review_count', 'N/A')}")
                
                # Show Blinkit-specific fields if available
                if platform == 'blinkit':
                    if product.get('brand'):
                        print(f"   Brand: {product.get('brand', 'N/A')}")
                    if product.get('variant'):
                        print(f"   Variant: {product.get('variant', 'N/A')}")
                    if product.get('eta'):
                        print(f"   ETA: {product.get('eta', 'N/A')}")
                print()
            
            if len(products) > 3:
                print(f"... and {len(products) - 3} more products")
            
            # Show output files
            if platform == 'blinkit':
                json_path = os.path.join(output_dir, "blinkit_data.json")
                csv_path = os.path.join(output_dir, "blinkit_data.csv")
            else:
                json_path = os.path.join(output_dir, "data.json")
                csv_path = os.path.join(output_dir, "data.csv")
            
            print_success(f"Results saved to:")
            print(f"  📄 JSON: {json_path}")
            print(f"  📊 CSV: {csv_path}")
                
        else:
            print_warning("No products found for the given search term.")
            print_warning("This could be due to:")
            if platform == 'blinkit':
                print_warning("  - Blinkit's API restrictions")
                print_warning("  - Network connectivity issues")
                print_warning("  - Changes in Blinkit's API structure")
                print_warning("  - The search term not returning results")
            else:
                print_warning("  - Amazon's anti-bot measures")
                print_warning("  - Network connectivity issues")
                print_warning("  - Changes in Amazon's HTML structure")
                print_warning("  - The search term not returning results")
    
    except KeyboardInterrupt:
        print_warning("Scraping interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print_error(f"An error occurred during scraping: {str(e)}")
        if args.verbose:
            import traceback
            print_error("Full traceback:")
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up scraper
        if 'scraper' in locals():
            scraper.session.close()
        
        print("\n" + "=" * 60)
        print("🏁 Scraping session completed")
        print("=" * 60)


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import requests
        import bs4
        return True
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_error("Please install requirements: pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Check if we have command line arguments
    if len(sys.argv) < 2:
        print_error("Please provide an item name to search for!")
        print("Usage: python main.py <item_name>")
        print("Example: python main.py oil")
        print("Example: python main.py \"body lotion\"")
        sys.exit(1)
    
    # Run the main function
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\nScraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
