"""
Interactive Blinkit scraper that takes user input for items and pincode
Enhanced version with user-friendly interface
"""

import json
import csv
import os
import time
import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from urllib.parse import quote_plus
import re
from bs4 import BeautifulSoup
import cloudscraper

# Try to import Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available. Install with: pip install selenium")


class BlinkitInteractiveScraper:
    """
    Interactive Blinkit scraper that takes user input for items and pincode.
    """
    
    def __init__(self):
        """Initialize the scraper."""
        self.setup_logging()
        self.driver = None
        self.pincode = None
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('blinkit_interactive_scraper.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_user_input(self):
        """Get user input for items and pincode."""
        print("🛒 Welcome to Blinkit Interactive Scraper!")
        print("=" * 50)
        
        # Get pincode
        while True:
            try:
                pincode = input("📍 Enter your pincode (6 digits): ").strip()
                if len(pincode) == 6 and pincode.isdigit():
                    self.pincode = pincode
                    break
                else:
                    print("❌ Please enter a valid 6-digit pincode")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                exit()
        
        # Get items to search
        print("\n🛍️ Enter items to search (separated by commas):")
        print("Example: milk, bread, eggs, curd, butter")
        
        while True:
            try:
                items_input = input("🔍 Items: ").strip()
                if items_input:
                    # Split by comma and clean up
                    items = [item.strip() for item in items_input.split(',') if item.strip()]
                    if items:
                        return items
                    else:
                        print("❌ Please enter at least one item")
                else:
                    print("❌ Please enter some items to search")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                exit()
    
    def setup_browser(self):
        """Setup Chrome browser with enhanced anti-detection."""
        if not SELENIUM_AVAILABLE:
            self.logger.error("❌ Selenium not available. Please install: pip install selenium")
            return False
        
        try:
            chrome_options = Options()
            
            # Enhanced anti-detection options
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--start-maximized")
            
            # Create driver
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.logger.info("✅ Interactive browser setup successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Interactive browser setup failed: {e}")
            return False
    
    def set_pincode(self, pincode: str):
        """Set pincode on Blinkit website and select the first location."""
        try:
            self.logger.info(f"📍 Setting pincode: {pincode}")
            
            # Navigate to Blinkit homepage
            self.driver.get("https://blinkit.com")
            time.sleep(3)
            
            # Look for pincode input field
            pincode_selectors = [
                "input[placeholder*='pincode']",
                "input[placeholder*='Pincode']",
                "input[placeholder*='PIN']",
                "input[placeholder*='pin']",
                "input[name*='pincode']",
                "input[name*='pin']",
                "input[type='text']",
                "input[class*='pincode']",
                "input[class*='pin']",
                "input[placeholder*='Enter']",
                "input[placeholder*='Search']",
            ]
            
            pincode_input = None
            for selector in pincode_selectors:
                try:
                    pincode_input = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if pincode_input.is_displayed():
                        break
                except:
                    continue
            
            if pincode_input:
                # Clear and enter pincode
                pincode_input.clear()
                pincode_input.send_keys(pincode)
                time.sleep(2)
                
                # Wait for location suggestions to appear
                self.logger.info("⏳ Waiting for location suggestions...")
                time.sleep(3)
                
                # Look for location suggestions/dropdown
                location_selectors = [
                    "[class*='suggestion']",
                    "[class*='dropdown']",
                    "[class*='option']",
                    "[class*='item']",
                    "[class*='location']",
                    "[class*='address']",
                    "li",
                    "div[role='option']",
                    "div[role='listbox'] li",
                    "[data-testid*='suggestion']",
                    "[data-testid*='option']",
                ]
                
                location_selected = False
                for selector in location_selectors:
                    try:
                        suggestions = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if suggestions:
                            self.logger.info(f"🔍 Found {len(suggestions)} location suggestions")
                            
                            # Click on the first suggestion
                            for suggestion in suggestions:
                                try:
                                    if suggestion.is_displayed() and suggestion.is_enabled():
                                        suggestion.click()
                                        self.logger.info("✅ Selected first location suggestion")
                                        location_selected = True
                                        break
                                except:
                                    continue
                            
                            if location_selected:
                                break
                    except:
                        continue
                
                # If no suggestions found, try pressing Enter
                if not location_selected:
                    self.logger.info("🔍 No suggestions found, trying Enter key...")
                    pincode_input.send_keys(Keys.ENTER)
                    time.sleep(2)
                
                # Wait for location to be set
                time.sleep(3)
                
                # Verify location is set by checking for location-specific elements
                if self.verify_location_set():
                    self.logger.info("✅ Pincode and location set successfully")
                    return True
                else:
                    self.logger.warning("⚠️ Location may not be set properly")
                    return True  # Continue anyway
            else:
                self.logger.warning("⚠️ Could not find pincode input field")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error setting pincode: {e}")
            return False
    
    def verify_location_set(self):
        """Verify that location is properly set by checking for location-specific elements."""
        try:
            # Look for elements that indicate location is set
            location_indicators = [
                "[class*='location']",
                "[class*='address']",
                "[class*='pincode']",
                "[class*='area']",
                "[class*='city']",
                "[data-testid*='location']",
                "[data-testid*='address']",
                "span[class*='location']",
                "div[class*='location']",
            ]
            
            for selector in location_indicators:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        for element in elements:
                            if element.is_displayed() and element.text.strip():
                                self.logger.info(f"✅ Location indicator found: {element.text.strip()}")
                                return True
                except:
                    continue
            
            # Check if we can see product-related elements (indicating location is set)
            product_indicators = [
                "[class*='product']",
                "[class*='item']",
                "[class*='card']",
                "[class*='snippet']",
            ]
            
            for selector in product_indicators:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements and len(elements) > 0:
                        self.logger.info(f"✅ Product elements found, location likely set")
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error verifying location: {e}")
            return False
    
    def scrape_item(self, item_name: str) -> List[Dict[str, Any]]:
        """Scrape a single item."""
        try:
            self.logger.info(f"🛒 Scraping: {item_name}")
            
            # Navigate to search page
            search_url = f"https://blinkit.com/s/?q={quote_plus(item_name)}"
            self.logger.info(f"🌐 Navigating to: {search_url}")
            
            self.driver.get(search_url)
            time.sleep(3)
            
            # Scroll down to load more products
            self.logger.info("📜 Scrolling to load more products...")
            for i in range(5):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # Wait for products to load
            self.logger.info("⏳ Waiting for products to load...")
            time.sleep(5)
            
            # Try to find product elements with multiple strategies
            products = []
            
            # Strategy 1: Wait for specific product selectors
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='product'], [class*='item'], [class*='card'], [data-testid*='product']"))
                )
                self.logger.info("✅ Product containers found!")
            except TimeoutException:
                self.logger.warning("⚠️ No product containers found, trying alternative approach...")
            
            # Strategy 2: Try multiple selectors for products
            product_selectors = [
                "[class*='product']",
                "[class*='item']",
                "[class*='card']",
                "[data-testid*='product']",
                "[class*='ProductCard']",
                "[class*='product-card']",
                "div[class*='snippet']",
                "div[class*='widget']",
                "div[class*='container']",
                "div[class*='grid']",
                "div[class*='list']",
                "div[class*='Product']",
                "div[class*='Item']",
                "div[class*='Card']",
                "div[class*='Snippet']",
                "div[class*='Widget']",
                "div[class*='Container']",
                "div[class*='Grid']",
                "div[class*='List']",
            ]
            
            all_elements = []
            for selector in product_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        self.logger.info(f"🔍 Found {len(elements)} elements with selector: {selector}")
                        all_elements.extend(elements)
                except Exception as e:
                    self.logger.warning(f"⚠️ Selector {selector} failed: {e}")
                    continue
            
            # Remove duplicates
            unique_elements = []
            seen_texts = set()
            for element in all_elements:
                try:
                    text = element.text.strip()
                    if text and text not in seen_texts and len(text) > 5:
                        unique_elements.append(element)
                        seen_texts.add(text)
                except:
                    continue
            
            self.logger.info(f"🔍 Found {len(unique_elements)} unique elements")
            
            # Extract products from elements
            for element in unique_elements:
                product = self.extract_product_from_element(element, item_name)
                if product and product.get('name'):
                    products.append(product)
            
            # Strategy 3: If still no products, try extracting from page source
            if not products:
                self.logger.info("🔍 No products found with selectors, trying page source extraction...")
                products = self.extract_products_from_page_source(item_name)
            
            self.logger.info(f"🛒 Found {len(products)} products for {item_name}")
            return products
            
        except Exception as e:
            self.logger.error(f"💥 Error scraping {item_name}: {e}")
            return []
    
    def extract_product_from_element(self, element, item_name: str) -> Optional[Dict[str, Any]]:
        """Extract product from a single element."""
        try:
            text = element.text.strip()
            if not text or len(text) < 5:
                return None
            
            # Check if the element contains the search term
            if item_name.lower() not in text.lower():
                return None
            
            # Extract price
            price_match = re.search(r'₹\s*(\d+)', text)
            price = price_match.group() if price_match else ""
            
            # Extract name
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            name = ""
            for line in lines:
                if len(line) > 3 and not re.search(r'₹\s*\d+', line):
                    if len(line) > 5 and len(line) < 100:
                        if item_name.lower() in line.lower():
                            name = line
                            break
            
            if not name:
                name = text[:50]
            
            return {
                'name': name,
                'price': price,
                'variant': '',
                'brand': '',
                'rating': '',
                'review_count': '',
                'product_url': '',
                'image_url': '',
                'product_id': '',
                'inventory': 0,
                'merchant_id': '',
                'merchant_type': '',
                'eta': '',
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting product from element: {e}")
            return None
    
    def extract_products_from_page_source(self, item_name: str) -> List[Dict[str, Any]]:
        """Extract products from page source."""
        products = []
        
        try:
            # Get page source
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Look for JSON data in script tags
            scripts = soup.find_all('script', type='application/json')
            scripts.extend(soup.find_all('script', type='application/ld+json'))
            scripts.extend(soup.find_all('script', string=re.compile(r'window\.__INITIAL_STATE__')))
            scripts.extend(soup.find_all('script', string=re.compile(r'window\.__NEXT_DATA__')))
            scripts.extend(soup.find_all('script', string=re.compile(r'window\.__APOLLO_STATE__')))
            scripts.extend(soup.find_all('script', string=re.compile(r'window\.__NUXT__')))
            
            for script in scripts:
                try:
                    if script.string:
                        data = json.loads(script.string)
                        if isinstance(data, dict):
                            products.extend(self.extract_products_from_json(data, item_name))
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    products.extend(self.extract_products_from_json(item, item_name))
                except:
                    continue
            
            # If no JSON products, try text extraction
            if not products:
                products = self.extract_products_from_text(soup.get_text(), item_name)
            
        except Exception as e:
            self.logger.error(f"Error extracting from page source: {e}")
        
        return products
    
    def extract_products_from_json(self, data: Dict[str, Any], item_name: str) -> List[Dict[str, Any]]:
        """Extract products from JSON data."""
        products = []
        
        try:
            # Look for common product data structures
            if 'products' in data:
                for product_data in data['products']:
                    product = self.parse_json_product(product_data, item_name)
                    if product:
                        products.append(product)
            
            if 'items' in data:
                for item_data in data['items']:
                    product = self.parse_json_product(item_data, item_name)
                    if product:
                        products.append(product)
            
            if 'data' in data and isinstance(data['data'], list):
                for item_data in data['data']:
                    product = self.parse_json_product(item_data, item_name)
                    if product:
                        products.append(product)
            
            # Look for nested structures
            for key, value in data.items():
                if isinstance(value, dict) and ('products' in value or 'items' in value):
                    products.extend(self.extract_products_from_json(value, item_name))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            products.extend(self.extract_products_from_json(item, item_name))
            
        except Exception as e:
            self.logger.warning(f"Error extracting products from JSON: {e}")
        
        return products
    
    def parse_json_product(self, product_data: Dict[str, Any], item_name: str) -> Optional[Dict[str, Any]]:
        """Parse a product from JSON data."""
        try:
            name = product_data.get('name', '') or product_data.get('title', '') or product_data.get('product_name', '')
            price = product_data.get('price', '') or product_data.get('selling_price', '') or product_data.get('mrp', '')
            
            if not name or len(name) < 3:
                return None
            
            # Check if the product is related to the search term
            if item_name.lower() not in name.lower():
                return None
            
            return {
                'name': name,
                'price': str(price) if price else '',
                'variant': product_data.get('variant', '') or product_data.get('size', '') or product_data.get('weight', ''),
                'brand': product_data.get('brand', '') or product_data.get('brand_name', ''),
                'rating': product_data.get('rating', ''),
                'review_count': product_data.get('review_count', '') or product_data.get('reviews', ''),
                'product_url': product_data.get('url', '') or product_data.get('product_url', ''),
                'image_url': product_data.get('image', '') or product_data.get('image_url', ''),
                'product_id': product_data.get('id', '') or product_data.get('product_id', ''),
                'inventory': product_data.get('inventory', 0) or product_data.get('stock', 0),
                'merchant_id': product_data.get('merchant_id', ''),
                'merchant_type': product_data.get('merchant_type', ''),
                'eta': product_data.get('eta', ''),
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Error parsing JSON product: {e}")
            return None
    
    def extract_products_from_text(self, text: str, item_name: str) -> List[Dict[str, Any]]:
        """Extract products from text."""
        products = []
        
        try:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            current_product = {}
            for i, line in enumerate(lines):
                # Look for price patterns
                price_match = re.search(r'₹\s*(\d+)', line)
                if price_match:
                    if current_product and current_product.get('name'):
                        products.append(current_product)
                    
                    current_product = {
                        'name': '',
                        'price': f"₹{price_match.group(1)}",
                        'variant': '',
                        'brand': '',
                        'rating': '',
                        'review_count': '',
                        'product_url': '',
                        'image_url': '',
                        'product_id': '',
                        'inventory': 0,
                        'merchant_id': '',
                        'merchant_type': '',
                        'eta': '',
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                    # Look for product name in nearby lines
                    for j in range(max(0, i-3), min(len(lines), i+3)):
                        if j != i and lines[j] and len(lines[j]) > 3:
                            if not re.search(r'₹\s*\d+', lines[j]) and not lines[j].isdigit():
                                if len(lines[j]) > 5 and len(lines[j]) < 100:
                                    # Check if the line contains the search term
                                    if item_name.lower() in lines[j].lower():
                                        current_product['name'] = lines[j]
                                        break
                
                elif current_product and not current_product.get('name') and len(line) > 3:
                    if not re.search(r'₹\s*\d+', line) and not line.isdigit():
                        if len(line) > 5 and len(line) < 100:
                            # Check if the line contains the search term
                            if item_name.lower() in line.lower():
                                current_product['name'] = line
            
            if current_product and current_product.get('name'):
                products.append(current_product)
            
        except Exception as e:
            self.logger.error(f"Error extracting from text: {e}")
        
        return products
    
    def save_results(self, all_products: Dict[str, List[Dict[str, Any]]], output_dir: str = "output"):
        """Save results to files."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save combined results
            combined_products = []
            for item, products in all_products.items():
                for product in products:
                    product['search_item'] = item
                    combined_products.append(product)
            
            # Save JSON
            json_path = os.path.join(output_dir, f"blinkit_interactive_products_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_products, f, indent=2, ensure_ascii=False)
            
            # Save CSV
            if combined_products:
                csv_path = os.path.join(output_dir, f"blinkit_interactive_products_{timestamp}.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=combined_products[0].keys())
                    writer.writeheader()
                    writer.writerows(combined_products)
            
            self.logger.info(f"💾 Saved {len(combined_products)} products to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def run_interactive_scraper(self):
        """Run the interactive scraper."""
        try:
            # Get user input
            items = self.get_user_input()
            
            print(f"\n🚀 Starting scraper for pincode: {self.pincode}")
            print(f"🛍️ Items to search: {', '.join(items)}")
            print("=" * 50)
            
            # Setup browser
            if not self.setup_browser():
                return
            
            # Set pincode
            self.set_pincode(self.pincode)
            
            # Scrape each item
            all_products = {}
            total_products = 0
            
            for i, item in enumerate(items, 1):
                print(f"\n🔍 [{i}/{len(items)}] Scraping: {item}")
                print("-" * 30)
                
                products = self.scrape_item(item)
                all_products[item] = products
                total_products += len(products)
                
                if products:
                    print(f"✅ Found {len(products)} products for {item}")
                    # Show sample products
                    for j, product in enumerate(products[:3]):
                        print(f"   {j+1}. {product.get('name', 'N/A')} - {product.get('price', 'N/A')}")
                else:
                    print(f"❌ No products found for {item}")
                
                # Add delay between searches
                if i < len(items):
                    time.sleep(3)
            
            # Save results
            self.save_results(all_products, "interactive_output")
            
            print(f"\n🎉 Scraping completed!")
            print(f"📊 Total products found: {total_products}")
            print(f"💾 Results saved to: interactive_output/")
            
        except KeyboardInterrupt:
            print("\n👋 Scraping interrupted by user")
        except Exception as e:
            print(f"❌ Error during scraping: {e}")
        finally:
            if self.driver:
                self.driver.quit()
                print("🔒 Browser closed")


def main():
    """Main function to run the interactive scraper."""
    scraper = BlinkitInteractiveScraper()
    scraper.run_interactive_scraper()


if __name__ == "__main__":
    main()
