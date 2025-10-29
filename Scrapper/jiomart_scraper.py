import requests
import json
import time
import random
import re
from datetime import datetime
from typing import List, Dict, Any


class JioMartScraper:
    def __init__(self):
        self.base_url = "https://www.jiomart.com"
        self.search_url = f"{self.base_url}/trex/search"
        self.session = requests.Session()
        self.setup_session()
    
    def setup_session(self):
        """Setup session with realistic headers"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'Origin': 'https://www.jiomart.com',
            'Referer': 'https://www.jiomart.com/search?q=milk',
            'Sec-Ch-Ua': '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Pragma': 'no-cache',
            'Priority': 'u=1, i'
        })
        
        # Add cookies from the network preview
        self.session.cookies.update({
            'nms_mgo_state_code': 'MH',
            '_ALGOLIA': 'anonymous-e5971e54-099e-46f2-94f7-50435298e2e3',
            '_fbp': 'fb.1.1761020147846.67791523',
            'WZRK_G': 'd30b8836e0184c7793fd7ec74732d0fd',
            '_gcl_au': '1.1.708694206.1761020150',
            '_ga': 'GA1.1.1766818152.1761020150',
            'nms_mgo_city': 'Pune',
            'nms_mgo_pincode': '411015',
            '_rsSource': 'qc',
            '_rsMedium': 'web',
            '_rsCampaign': 'price_compare',
            '_rsUTMtrackingsource': 'qc%26web%26price_compare',
            'AKA_A2': 'A'
        })
    
    def search_products(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for products on JioMart"""
        try:
            # print(f"JioMart: Searching for '{query}'...")  # Removed INFO log
            
            # Try API search
            results = self._try_api_search(query, max_results)
            if results:
                # print(f"JioMart: Found {len(results)} products via API")  # Removed INFO log
                return results
            
            # print(f"JioMart: No products found for '{query}'")  # Removed INFO log
            return []
            
        except Exception as e:
            print(f"JioMart: Error searching '{query}': {e}")
            return []
    
    def _try_api_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Try JioMart API search"""
        try:
            # Try a minimal payload with required fields
            search_payload = {
                "query": query,
                "visitorId": "anonymous-e5971e54-099e-46f2-94f7-50435298e2e3"
            }
            
            # print(f"Trying JioMart API: {self.search_url}")  # Removed INFO log
            
            response = self.session.post(self.search_url, json=search_payload, timeout=15)
            
            # print(f"JioMart API status: {response.status_code}")  # Removed INFO log
            
            if response.status_code == 200:
                data = response.json()
                # print(f"JioMart API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")  # Removed INFO log
                # print(f"First result structure: {data['results'][0] if data.get('results') else 'No results'}")  # Removed INFO log
                products = self._parse_api_response(data, max_results)
                if products:
                    return products
            elif response.status_code == 403:
                print("JioMart API blocked with 403 - anti-bot protection")
            else:
                print(f"JioMart API returned status code: {response.status_code}")
                print(f"Response text: {response.text[:200]}")
            
            return []
                
        except Exception as e:
            print(f"Error calling JioMart API: {e}")
            return []
    
    def _parse_api_response(self, data: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Parse JioMart API response"""
        products = []
        
        try:
            # Extract products from the API response
            if 'results' in data and isinstance(data['results'], list):
                product_list = data['results']
                print(f"Found {len(product_list)} products in JioMart API response")
                
                for product in product_list[:max_results]:
                    parsed_product = self._extract_product_from_api(product)
                    if parsed_product:
                        products.append(parsed_product)
            else:
                print("No results found in JioMart API response")
                return []
                
        except Exception as e:
            print(f"Error parsing JioMart API response: {e}")
        
        return products
    
    def _extract_product_from_api(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product information from JioMart API product data"""
        try:
            # Extract basic product information from the actual structure
            product_info = product.get('product', {})
            name = product_info.get('title', '')
            
            # Extract brand information
            brand = ''
            if 'brands' in product_info and isinstance(product_info['brands'], list) and len(product_info['brands']) > 0:
                brand = product_info['brands'][0]
            
            # Extract pricing information from attributes
            attributes = product_info.get('attributes', {})
            price = ''
            original_price = ''
            
            if 'avg_selling_price' in attributes:
                price_data = attributes['avg_selling_price']
                if 'numbers' in price_data and len(price_data['numbers']) > 0:
                    price = str(price_data['numbers'][0])
            
            # Extract MRP from buybox_mrp
            if 'buybox_mrp' in attributes:
                mrp_data = attributes['buybox_mrp']
                if 'text' in mrp_data and len(mrp_data['text']) > 0:
                    # Parse the MRP string to extract price
                    mrp_string = mrp_data['text'][0]
                    # Extract price from string like "QCPANINDIAGROCERIES|1|Reliance Retail||28.0|28.0|||||1|"
                    price_match = re.search(r'(\d+\.?\d*)\|', mrp_string)
                    if price_match:
                        original_price = price_match.group(1)
            
            # Extract image URL
            image_url = ''
            images = product_info.get('images', [])
            if images and len(images) > 0:
                image_url = images[0].get('uri', '') if isinstance(images[0], dict) else str(images[0])
            
            # Extract product URL
            product_url = product.get('uri', '')
            
            # Extract categories
            categories = product_info.get('categories', [])
            category = categories[0] if categories else ''
            
            # Calculate discount percentage
            discount = ''
            if price and original_price and price != original_price:
                try:
                    price_val = float(price)
                    original_val = float(original_price)
                    if original_val > 0:
                        discount_percent = ((original_val - price_val) / original_val) * 100
                        discount = f"{discount_percent:.0f}% off"
                except:
                    pass
            
            if name and price:
                return {
                    'name': name,
                    'price': price,
                    'original_price': original_price,
                    'brand': brand,
                    'variant': '',
                    'image_url': image_url,
                    'product_url': product_url,
                    'rating': '',
                    'review_count': '',
                    'available': True,
                    'eta': '',
                    'discount': discount,
                    'category': category,
                    'scraped_at': datetime.now().isoformat(),
                    'platform': 'jiomart'
                }
                
        except Exception as e:
            print(f"Error extracting product from JioMart API: {e}")
        
        return None


if __name__ == "__main__":
    scraper = JioMartScraper()
    products = scraper.search_products('milk', 3)
    print(f"Found {len(products)} products")
    for p in products:
        print(f"- {p['name']}: {p['price']} by {p['brand']}")
