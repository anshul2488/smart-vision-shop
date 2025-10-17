"""
Blinkit scraper using requests and JSON API
"""

import json
import csv
import os
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
import logging

try:
    from .utils import format_output_data, validate_url, build_blinkit_url, build_blinkit_alternative_urls
except ImportError:
    from utils import format_output_data, validate_url, build_blinkit_url, build_blinkit_alternative_urls


class BlinkitScraper:
    """
    Blinkit scraper using requests and JSON API for product listings.
    """
    
    def __init__(self):
        """Initialize the Blinkit scraper."""
        self.session = None
        self.setup_logging()
        self.setup_session()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_session(self):
        """Setup requests session with headers."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Connection': 'keep-alive',
            'Origin': 'https://blinkit.com',
            'Referer': 'https://blinkit.com/s/?q=curd&limit=50&page=1',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Content-Type': 'application/json',
            'app_client': 'consumer_web',
            'app_version': '1010101010',
            'auth_key': 'c761ec3633c22afad934fb17a66385c1c06c5472b4898b866b7306186d0bb477',
            'device_id': '2ae5f67b6056cbe9',
            'lat': '18.5534027',
            'lon': '73.75321890000001',
            'rn_bundle_version': '1009003012',
            'web_app_version': '1008010016',
            'access_token': 'null',
            'session_uuid': 'c710429f-b71d-4a54-a529-177ae6f5dcca',
        })
    
    def fetch_api_data(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch API data using requests.
        
        Args:
            url (str): API URL to fetch
            
        Returns:
            Optional[Dict[str, Any]]: JSON response or None if failed
        """
        try:
            self.logger.info(f"Fetching API data: {url}")
            
            # Add random delay to be respectful
            time.sleep(random.uniform(1, 3))
            
            # Fetch the API data using POST method (as per the real API call)
            response = self.session.post(url, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            self.logger.info("API data fetched successfully")
            return data
                
        except Exception as e:
            self.logger.error(f"Error fetching API data: {e}")
            return None
    
    def parse_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse API response to extract product information.
        
        Args:
            data (Dict[str, Any]): API response data
            
        Returns:
            List[Dict[str, Any]]: List of product dictionaries
        """
        products = []
        
        try:
            # Check if the response is successful
            if not data.get('is_success', False):
                self.logger.warning("API response indicates failure")
                return products
            
            # Extract snippets from response
            response_data = data.get('response', {})
            snippets = response_data.get('snippets', [])
            
            if not snippets:
                self.logger.warning("No product snippets found in response")
                return products
            
            self.logger.info(f"Found {len(snippets)} product snippets")
            
            for snippet in snippets:
                try:
                    product = self._extract_product_info(snippet)
                    if product:
                        products.append(product)
                except Exception as e:
                    self.logger.warning(f"Error extracting product info: {e}")
                    continue
            
            self.logger.info(f"Successfully parsed {len(products)} products")
            
        except Exception as e:
            self.logger.error(f"Error parsing API response: {e}")
        
        return products
    
    def _extract_product_info(self, snippet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract product information from a single snippet.
        
        Args:
            snippet (Dict[str, Any]): Product snippet data
            
        Returns:
            Optional[Dict[str, Any]]: Product information dictionary
        """
        try:
            data = snippet.get('data', {})
            
            # Extract product name
            name_data = data.get('name', {})
            name = name_data.get('text', '') if name_data else ''
            
            if not name:
                # Try alternative name fields
                name = data.get('display_name', {}).get('text', '') if data.get('display_name') else ''
            
            if not name:
                return None
            
            # Extract price
            price_data = data.get('normal_price', {})
            price = price_data.get('text', '') if price_data else ''
            
            # Extract variant/size
            variant_data = data.get('variant', {})
            variant = variant_data.get('text', '') if variant_data else ''
            
            # Extract brand
            brand = data.get('brand_name', {}).get('text', '') if data.get('brand_name') else ''
            
            # Extract rating
            rating_data = data.get('rating', {})
            rating = ""
            review_count = ""
            
            if rating_data and rating_data.get('type') == 'bar':
                bar_data = rating_data.get('bar', {})
                rating_value = bar_data.get('value', 0)
                if rating_value:
                    rating = str(rating_value)
                
                # Extract review count from title
                title_data = bar_data.get('title', {})
                if title_data:
                    review_text = title_data.get('text', '')
                    # Extract number from text like "(3,219)"
                    import re
                    review_match = re.search(r'\(([\d,]+)\)', review_text)
                    if review_match:
                        review_count = review_match.group(1)
            
            # Extract image URL
            image_data = data.get('image', {})
            image_url = image_data.get('url', '') if image_data else ''
            
            # Extract product ID
            identity = data.get('identity', {})
            product_id = identity.get('id', '') if identity else ''
            
            # Extract inventory
            inventory = data.get('inventory', 0)
            
            # Extract merchant info
            merchant_id = data.get('merchant_id', '')
            merchant_type = data.get('merchant_type', '')
            
            # Extract ETA info
            eta_tag = data.get('eta_tag', {})
            eta_text = ""
            if eta_tag:
                eta_title = eta_tag.get('title', {})
                eta_text = eta_title.get('text', '') if eta_title else ''
            
            # Build product URL (Blinkit deeplink)
            product_url = ""
            click_action = data.get('click_action', {})
            if click_action and click_action.get('type') == 'blinkit_deeplink':
                deeplink = click_action.get('blinkit_deeplink', {})
                product_url = deeplink.get('url', '')
            
            return {
                'name': name,
                'price': price,
                'variant': variant,
                'brand': brand,
                'rating': rating,
                'review_count': review_count,
                'product_url': product_url,
                'image_url': image_url,
                'product_id': product_id,
                'inventory': inventory,
                'merchant_id': merchant_id,
                'merchant_type': merchant_type,
                'eta': eta_text,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting product info: {e}")
            return None
    
    def _scrape_blinkit_data_only(self, item_name: str) -> List[Dict[str, Any]]:
        """
        Scrape Blinkit data without saving to files (for use in price scraper)
        
        Args:
            item_name: Name of the item to search for
            
        Returns:
            List of product dictionaries
        """
        try:
            # Build the search URL
            search_url = build_blinkit_url(item_name)
            self.logger.info(f"Searching Blinkit for: {item_name}")
            self.logger.info(f"URL: {search_url}")
            
            # Fetch data from the API
            data = self.fetch_api_data(search_url)
            
            if not data:
                self.logger.warning("No data received from Blinkit API")
                return []
            
            # Parse the results
            products = self.parse_results(data)
            
            if not products:
                self.logger.warning("No products found in the response")
                return []
            
            self.logger.info(f"Found {len(products)} products")
            return products
            
        except Exception as e:
            self.logger.error(f"Error scraping Blinkit data: {e}")
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
            json_path = os.path.join(output_dir, "blinkit_data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Data saved to JSON: {json_path}")
            
            # Save as CSV
            if formatted_data:
                csv_path = os.path.join(output_dir, "blinkit_data.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=formatted_data[0].keys())
                    writer.writeheader()
                    writer.writerows(formatted_data)
                
                self.logger.info(f"Data saved to CSV: {csv_path}")
            
            self.logger.info(f"Successfully saved {len(formatted_data)} products")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def scrape_blinkit(self, item_name: str, output_dir: str = "output") -> List[Dict[str, Any]]:
        """
        Main method to scrape Blinkit for a given item.
        
        Args:
            item_name (str): Item to search for
            output_dir (str): Output directory for results
            
        Returns:
            List[Dict[str, Any]]: Scraped product data
        """
        try:
            # Build the Blinkit API URL
            url = build_blinkit_url(item_name)
            self.logger.info(f"Searching for: {item_name}")
            self.logger.info(f"Primary API URL: {url}")
            
            # Try the primary URL first
            api_data = self.fetch_api_data(url)
            
            # If primary URL fails, try alternative URLs
            if not api_data:
                self.logger.warning("Primary URL failed, trying alternative endpoints...")
                alternative_urls = build_blinkit_alternative_urls(item_name)
                
                for alt_url in alternative_urls:
                    self.logger.info(f"Trying alternative URL: {alt_url}")
                    api_data = self.fetch_api_data(alt_url)
                    if api_data:
                        self.logger.info(f"Success with alternative URL: {alt_url}")
                        break
                    time.sleep(1)  # Small delay between attempts
            
            if not api_data:
                self.logger.error("Failed to fetch API data from all endpoints")
                return []
            
            # Parse the results
            products = self.parse_results(api_data)
            if not products:
                self.logger.warning("No products found")
                return []
            
            # Save the results
            self.save_results(products, output_dir)
            
            return products
            
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
            raise
        finally:
            # Clean up session
            if self.session:
                self.session.close()
