"""
Amazon India scraper using requests and BeautifulSoup
"""

import json
import csv
import os
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import logging

try:
    from .utils import format_output_data, validate_url, build_amazon_url
except ImportError:
    from utils import format_output_data, validate_url, build_amazon_url


class AmazonScraper:
    """
    Amazon India scraper using requests and BeautifulSoup for product listings.
    """
    
    def __init__(self):
        """Initialize the Amazon scraper."""
        self.session = None
        self.setup_logging()
        self.setup_session()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_session(self):
        """Setup requests session with headers."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"'
        })
    
    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch page content using requests.
        
        Args:
            url (str): URL to fetch
            
        Returns:
            Optional[str]: HTML content or None if failed
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Fetching page
                
                # Removed delay for faster processing
                
                # Update referer for subsequent requests
                if attempt > 0:
                    self.session.headers.update({
                        'Referer': 'https://www.amazon.in/',
                        'Origin': 'https://www.amazon.in'
                    })
                
                # Fetch the page
                response = self.session.get(url, timeout=30)
                
                # Check for specific error codes
                if response.status_code == 503:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return None
                
                response.raise_for_status()
                
                # Check if we got a valid HTML response
                if 'html' in response.headers.get('content-type', '').lower():
                    return response.text
                else:
                    return None
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return None
                continue
        
        return None
    
    def parse_results(self, html: str) -> List[Dict[str, Any]]:
        """
        Parse HTML content to extract product information.
        
        Args:
            html (str): HTML content to parse
            
        Returns:
            List[Dict[str, Any]]: List of product dictionaries
        """
        products = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find product containers - Amazon uses different selectors
            product_selectors = [
                '[data-component-type="s-search-result"]',
                '.s-result-item',
                '[data-asin]'
            ]
            
            product_containers = []
            for selector in product_selectors:
                containers = soup.select(selector)
                if containers:
                    product_containers = containers
                    break
            
            if not product_containers:
                return products
            
            for container in product_containers:
                try:
                    product = self._extract_product_info(container)
                    if product:
                        products.append(product)
                except Exception as e:
                    continue
            
            # Successfully parsed products
            
        except Exception as e:
            pass
        
        return products
    
    def _extract_product_info(self, container) -> Optional[Dict[str, Any]]:
        """
        Extract product information from a single product container.
        
        Args:
            container: BeautifulSoup element containing product info
            
        Returns:
            Optional[Dict[str, Any]]: Product information dictionary
        """
        try:
            # Extract product name
            name_selectors = [
                'h2 a span',
                '.s-size-mini .s-color-base',
                '[data-cy="title-recipe-title"]',
                'h2 span',
                '.s-title-instructions-style'
            ]
            
            name = ""
            for selector in name_selectors:
                name_elem = container.select_one(selector)
                if name_elem:
                    name = name_elem.get_text(strip=True)
                    break
            
            if not name:
                return None
            
            # Extract price
            price_selectors = [
                '.a-price-whole',
                '.a-price .a-offscreen',
                '.a-price-range',
                '.a-price-symbol + span',
                '[data-a-color="price"] .a-offscreen'
            ]
            
            price = ""
            for selector in price_selectors:
                price_elem = container.select_one(selector)
                if price_elem:
                    price = price_elem.get_text(strip=True)
                    break
            
            # Extract rating
            rating_selectors = [
                '.a-icon-alt',
                '[data-cy="rating-recipe"] .a-icon-alt',
                '.a-icon-star-small .a-icon-alt'
            ]
            
            rating = ""
            for selector in rating_selectors:
                rating_elem = container.select_one(selector)
                if rating_elem:
                    rating = rating_elem.get_text(strip=True)
                    break
            
            # Extract review count
            review_selectors = [
                '.a-size-base',
                '[data-cy="rating-recipe"] .a-size-base',
                '.a-link-normal .a-size-base'
            ]
            
            review_count = ""
            for selector in review_selectors:
                review_elem = container.select_one(selector)
                if review_elem and 'review' in review_elem.get_text().lower():
                    review_count = review_elem.get_text(strip=True)
                    break
            
            # Extract product URL
            product_url = ""
            link_elem = container.select_one('h2 a, .s-title-instructions-style a')
            if link_elem and link_elem.get('href'):
                href = link_elem.get('href')
                if href.startswith('/'):
                    product_url = f"https://www.amazon.in{href}"
                else:
                    product_url = href
            
            # Extract image URL
            image_url = ""
            img_elem = container.select_one('img')
            if img_elem and img_elem.get('src'):
                image_url = img_elem.get('src')
            elif img_elem and img_elem.get('data-src'):
                image_url = img_elem.get('data-src')
            
            return {
                'name': name,
                'price': price,
                'rating': rating,
                'review_count': review_count,
                'product_url': product_url,
                'image_url': image_url,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return None
    
    def _search_products_data_only(self, item_name: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Amazon products without saving to files (for use in price scraper)
        
        Args:
            item_name: Name of the item to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of product dictionaries
        """
        try:
            # Build the search URL
            search_url = build_amazon_url(item_name)
            
            # Fetch the HTML content
            html_content = self.fetch_page(search_url)
            
            if not html_content:
                return []
            
            # Parse the HTML and extract product information
            products = self.parse_results(html_content)
            
            if not products:
                return []
            
            # Found products
            return products
            
        except Exception as e:
            return []
    
    def save_results(self, data: List[Dict[str, Any]], output_dir: str = "output"):
        """
        Save scraped data to JSON and CSV files.
        
        Args:
            data (List[Dict[str, Any]]): Product data to save
            output_dir (str): Output directory path
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Format the data
            formatted_data = format_output_data(data)
            
            # Save as JSON
            json_path = os.path.join(output_dir, "data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, indent=2, ensure_ascii=False)
            
            # Save as CSV
            if formatted_data:
                csv_path = os.path.join(output_dir, "data.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=formatted_data[0].keys())
                    writer.writeheader()
                    writer.writerows(formatted_data)
            
        except Exception as e:
            pass
    
    def search_products(self, item_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for products on Amazon (wrapper for price scraper compatibility)
        
        Args:
            item_name: Name of the item to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of product dictionaries
        """
        try:
            print(f"🔍 Amazon: Searching for '{item_name}'...")
            results = self._search_products_data_only(item_name, max_results)
            if results:
                print(f"✅ Amazon: Found {len(results)} products")
                return results
            else:
                print(f"❌ Amazon: No products found for '{item_name}'")
                return []
        except Exception as e:
            print(f"❌ Amazon: Error searching '{item_name}': {e}")
            return []

    def scrape_amazon(self, item_name: str, output_dir: str = "output") -> List[Dict[str, Any]]:
        """
        Main method to scrape Amazon for a given item.
        
        Args:
            item_name (str): Item to search for
            output_dir (str): Output directory for results
            
        Returns:
            List[Dict[str, Any]]: Scraped product data
        """
        try:
            # Build the Amazon URL
            url = build_amazon_url(item_name)
            
            # Fetch the page
            html = self.fetch_page(url)
            if not html:
                return []
            
            # Parse the results
            products = self.parse_results(html)
            if not products:
                return []
            
            # Save the results
            self.save_results(products, output_dir)
            
            return products
            
        except Exception as e:
            return []
        finally:
            # Clean up session
            if self.session:
                self.session.close()
