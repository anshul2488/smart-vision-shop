import requests
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup


class BigBasketScraper:
    def __init__(self):
        self.base_url = "https://www.bigbasket.com"
        self.search_url = f"{self.base_url}/ps"
        self.session = requests.Session()
        self.setup_session()
    
    def setup_session(self):
        """Setup session with realistic headers"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
    
    def search_products(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for products on BigBasket"""
        try:
            print(f"BigBasket: Searching for '{query}'...")
            
            # Try API first
            results = self._try_api_search(query, max_results)
            if results:
                print(f"BigBasket: Found {len(results)} products via API")
                return results
            
            # Try web scraping as fallback
            print("BigBasket API failed, trying web scraping...")
            results = self._try_web_scraping(query, max_results)
            if results:
                print(f"BigBasket: Found {len(results)} products via web scraping")
                return results
            
            print(f"BigBasket: No products found for '{query}'")
            return []
            
        except Exception as e:
            print(f"BigBasket: Error searching '{query}': {e}")
            return []
    
    def _try_api_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Try BigBasket API search"""
        try:
            # Try mobile API endpoint
            mobile_api_url = "https://m.bigbasket.com/api/search"
            
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.9',
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': 'https://m.bigbasket.com/',
                'Origin': 'https://m.bigbasket.com'
            }
            
            params = {
                'q': query,
                'limit': max_results,
                'page': 1
            }
            
            print(f"Trying mobile API: {mobile_api_url}")
            
            response = requests.get(mobile_api_url, params=params, headers=headers, timeout=15)
            
            print(f"Mobile API status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Mobile API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                products = self._parse_api_response(data, max_results)
                if products:
                    return products
            
            # Try main API endpoint
            return self._try_main_api(query, max_results)
                
        except Exception as e:
            print(f"Error calling mobile API: {e}")
            return self._try_main_api(query, max_results)
    
    def _try_main_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Try main BigBasket API"""
        try:
            api_url = "https://www.bigbasket.com/listing-svc/v2/products"
            
            # First visit main page
            main_url = f"https://www.bigbasket.com/ps/?q={query}"
            main_response = requests.get(main_url, headers=self.session.headers, timeout=15)
            print(f"Main page status: {main_response.status_code}")
            
            # Wait to simulate human behavior
            time.sleep(random.uniform(2, 4))
            
            # Try API
            params = {
                'type': 'ps',
                'slug': query,
                'page': '1',
                'bucket_id': '4'
            }
            
            headers = {
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
                'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
                'Cache-Control': 'no-cache',
                'Content-Type': 'application/json',
                'Referer': f'https://www.bigbasket.com/ps/?q={query}&nc=as',
                'Sec-Ch-Ua': '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
                'X-Channel': 'BB-WEB',
                'X-Entry-Context': 'bbnow',
                'X-Entry-Context-Id': '10',
                'X-Integrated-Fc-Door-Visible': 'false',
                'X-Tracker': '32e1d7c6-a4fe-478c-8574-da595b3b6628'
            }
            
            response = requests.get(api_url, params=params, headers=headers, timeout=15)
            
            print(f"Main API status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Main API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                products = self._parse_api_response(data, max_results)
                if products:
                    return products
            elif response.status_code == 403:
                print("Main API blocked with 403 - anti-bot protection")
            else:
                print(f"Main API returned status code: {response.status_code}")
            
            return []
                
        except Exception as e:
            print(f"Error calling main API: {e}")
            return []
    
    def _parse_api_response(self, data: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Parse API response data"""
        products = []
        
        try:
            # Try different response structures
            if 'tabs' in data and len(data['tabs']) > 0:
                # BigBasket API structure
                first_tab = data['tabs'][0]
                if 'product_info' in first_tab and 'products' in first_tab['product_info']:
                    product_list = first_tab['product_info']['products']
                else:
                    return []
            elif 'data' in data and 'products' in data['data']:
                product_list = data['data']['products']
            elif 'products' in data:
                product_list = data['products']
            elif 'items' in data:
                product_list = data['items']
            elif 'results' in data:
                product_list = data['results']
            elif isinstance(data, list):
                product_list = data
            else:
                return []
            
            for product in product_list[:max_results]:
                parsed = self._parse_product(product)
                if parsed:
                    products.append(parsed)
                    
        except Exception as e:
            print(f"Error parsing API response: {e}")
        
        return products
    
    def _parse_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Parse individual product data"""
        try:
            # Extract product information
            name = product.get('desc', product.get('name', product.get('title', '')))
            brand = ''
            if 'brand' in product and isinstance(product['brand'], dict):
                brand = product['brand'].get('name', '')
            elif 'brand' in product:
                brand = str(product['brand'])
            
            # Extract pricing
            price = ''
            original_price = ''
            if 'pricing' in product and 'discount' in product['pricing']:
                discount_info = product['pricing']['discount']
                price = discount_info.get('prim_price', {}).get('sp', '')
                original_price = discount_info.get('mrp', '')
            elif 'price' in product:
                price = str(product['price'])
                original_price = price
            
            # Extract other info
            variant = product.get('w', product.get('variant', product.get('size', '')))
            image_url = ''
            if 'images' in product and len(product['images']) > 0:
                image_url = product['images'][0].get('m', product['images'][0].get('url', ''))
            
            product_url = product.get('absolute_url', product.get('url', ''))
            if product_url and not product_url.startswith('http'):
                product_url = 'https://www.bigbasket.com' + product_url
            
            eta = ''
            if 'availability' in product:
                eta = product['availability'].get('short_eta', '')
            
            if name and price:
                return {
                    'name': name,
                    'price': price,
                    'original_price': original_price,
                    'brand': brand,
                    'variant': variant,
                    'image_url': image_url,
                    'product_url': product_url,
                    'rating': '',
                    'review_count': '',
                    'available': True,
                    'eta': eta,
                    'discount': '',
                    'scraped_at': datetime.now().isoformat(),
                    'platform': 'bigbasket'
                }
                
        except Exception as e:
            print(f"Error parsing product: {e}")
        
        return None
    
    def _try_web_scraping(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Try web scraping approach"""
        try:
            search_url = f"https://www.bigbasket.com/ps/?q={query.replace(' ', '+')}"
            
            print(f"Trying web scraping from: {search_url}")
            
            response = requests.get(search_url, headers=self.session.headers, timeout=15)
            
            if response.status_code == 200:
                print(f"Web scraping response status: {response.status_code}")
                return self._parse_html_products(response.text, query, max_results)
            else:
                print(f"Web scraping failed with status: {response.status_code}")
                
        except Exception as e:
            print(f"Error in web scraping: {e}")
        
        return []
    
    def _parse_html_products(self, html_content: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse HTML content to extract product information"""
        products = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for JSON data in script tags
            script_tags = soup.find_all('script', type='application/json')
            
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        products.extend(self._parse_api_response(data, max_results))
                        if products:
                            break
                except:
                    continue
            
            # If no JSON data found, try HTML parsing
            if not products:
                products = self._parse_html_product_cards(soup, max_results)
            
            print(f"Extracted {len(products)} products from HTML")
            
        except Exception as e:
            print(f"Error parsing HTML: {e}")
        
        return products
    
    def _parse_html_product_cards(self, soup, max_results: int) -> List[Dict[str, Any]]:
        """Parse product cards from HTML"""
        products = []
        
        try:
            # Look for product containers
            product_selectors = [
                '.product-item',
                '.product-card',
                '.item-card',
                '.product',
                '[data-testid*="product"]'
            ]
            
            product_elements = []
            for selector in product_selectors:
                elements = soup.select(selector)
                if elements:
                    product_elements.extend(elements)
                    break
            
            for element in product_elements[:max_results]:
                try:
                    product = self._extract_product_from_element(element)
                    if product:
                        products.append(product)
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error parsing HTML product cards: {e}")
        
        return products
    
    def _extract_product_from_element(self, element) -> Dict[str, Any]:
        """Extract product information from HTML element"""
        try:
            # Extract product name
            name_selectors = ['.product-name', '.item-name', '.product-title', 'h3', 'h4', 'h5']
            name = ""
            for selector in name_selectors:
                name_elem = element.select_one(selector)
                if name_elem and name_elem.get_text(strip=True):
                    name = name_elem.get_text(strip=True)
                    break
            
            # Extract price
            price_selectors = ['.price', '.product-price', '.item-price']
            price = ""
            for selector in price_selectors:
                price_elem = element.select_one(selector)
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    import re
                    price_match = re.search(r'[\d,]+', price_text)
                    if price_match:
                        price = price_match.group().replace(',', '')
                        break
            
            # Extract image URL
            img_elem = element.select_one('img')
            image_url = img_elem.get('src', '') if img_elem else ''
            
            # Extract product URL
            link_elem = element.select_one('a')
            product_url = link_elem.get('href', '') if link_elem else ''
            if product_url and not product_url.startswith('http'):
                product_url = 'https://www.bigbasket.com' + product_url
            
            if name and price:
                return {
                    'name': name,
                    'price': price,
                    'original_price': price,
                    'brand': '',
                    'variant': '',
                    'image_url': image_url,
                    'product_url': product_url,
                    'rating': '',
                    'review_count': '',
                    'available': True,
                    'eta': '',
                    'discount': '',
                    'scraped_at': datetime.now().isoformat(),
                    'platform': 'bigbasket'
                }
                
        except Exception as e:
            pass
        
        return None


if __name__ == "__main__":
    scraper = BigBasketScraper()
    products = scraper.search_products('bread', 3)
    print(f"Found {len(products)} products")
    for p in products:
        print(f"- {p['name']}: {p['price']} by {p['brand']}")
