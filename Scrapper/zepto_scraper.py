"""
Zepto Scraper for Grocery Items
Uses Zepto's API to search for products and get prices
"""
import requests
import json
import time
import uuid
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import hmac
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ZeptoScraper:
    """Scraper for Zepto grocery delivery platform"""
    
    def __init__(self):
        self.base_url = "https://cdn.bff.zeptonow.com/api/v3"
        self.search_endpoint = f"{self.base_url}/search"
        self.session = requests.Session()
        
        # Alternative endpoints to try
        self.alternative_endpoints = [
            "https://api.zeptonow.com/v1/search",
            "https://www.zeptonow.com/api/v1/search",
            "https://cdn.bff.zeptonow.com/api/v2/search"
        ]
        
        # Configure session with retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set up session with realistic browser behavior
        self._setup_session()
        
        # Default headers based on the provided information
        self.default_headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
            'app_sub_platform': 'WEB',
            'app_version': '13.28.1',
            'appversion': '13.28.1',
            'auth_revamp_flow': 'v2',
            'compatible_components': 'CONVENIENCE_FEE,RAIN_FEE,EXTERNAL_COUPONS,STANDSTILL,BUNDLE,MULTI_SELLER_ENABLED,PIP_V1,ROLLUPS,SCHEDULED_DELIVERY,SAMPLING_ENABLED,ETA_NORMAL_WITH_149_DELIVERY,ETA_NORMAL_WITH_199_DELIVERY,HOMEPAGE_V2,NEW_ETA_BANNER,VERTICAL_FEED_PRODUCT_GRID,AUTOSUGGESTION_PAGE_ENABLED,AUTOSUGGESTION_PIP,AUTOSUGGESTION_AD_PIP,BOTTOM_NAV_FULL_ICON,COUPON_WIDGET_CART_REVAMP,DELIVERY_UPSELLING_WIDGET,MARKETPLACE_CATEGORY_GRID,NO_PLATFORM_CHECK_ENABLED_V2,SUPER_SAVER:1,SUPERSTORE_V1,PROMO_CASH:0,24X7_ENABLED_V1,TABBED_CAROUSEL_V2,HP_V4_FEED,WIDGET_BASED_ETA,PC_REVAMP_1,NO_COST_EMI_V1,PRE_SEARCH,ITEMISATION_ENABLED,ZEPTO_PASS,ZEPTO_PASS:5,BACHAT_FOR_ALL,SAMPLING_UPSELL_CAMPAIGN,DISCOUNTED_ADDONS_ENABLED,UPSELL_COUPON_SS:0,NEW_ROLLUPS_ENABLED,RERANKING_QCL_RELATED_PRODUCTS,PLP_ON_SEARCH,PAAN_BANNER_WIDGETIZED,ROLLUPS_UOM,DYNAMIC_FILTERS,PHARMA_ENABLED,AUTOSUGGESTION_RECIPE_PIP,SEARCH_FILTERS_V1,QUERY_DESCRIPTION_WIDGET,MEDS_WITH_SIMILAR_SALT_WIDGET,NEW_FEE_STRUCTURE,NEW_BILL_INFO,RE_PROMISE_ETA_ORDER_SCREEN_ENABLED,SUPERSTORE_V1,MANUALLY_APPLIED_DELIVERY_FEE_RECEIVABLE,MARKETPLACE_REPLACEMENT,ZEPTO_PASS,ZEPTO_PASS:5,ZEPTO_PASS_RENEWAL,CART_REDESIGN_ENABLED,SHIPMENT_WIDGETIZATION_ENABLED,TABBED_CAROUSEL_V2,24X7_ENABLED_V1,PROMO_CASH:0,HOMEPAGE_V2,SUPER_SAVER:1,NO_PLATFORM_CHECK_ENABLED_V2,HP_V4_FEED,GIFT_CARD,SCLP_ADD_MONEY,GIFTING_ENABLED,OFSE,WIDGET_BASED_ETA,PC_REVAMP_1,NEW_ETA_BANNER,NO_COST_EMI_V1,ITEMISATION_ENABLED,SWAP_AND_SAVE_ON_CART,WIDGET_RESTRUCTURE,PRICING_CAMPAIGN_ID,BACHAT_FOR_ALL,TABBED_CAROUSEL_V3,CART_LMS:2,SAMPLING_UPSELL_CAMPAIGN,DISCOUNTED_ADDONS_ENABLED,UPSELL_COUPON_SS:0,SIZE_EXCHANGE_ENABLED',
            'content-type': 'application/json',
            'marketplace_type': 'ZEPTO_NOW',
            'origin': 'https://www.zeptonow.com',
            'platform': 'WEB',
            'priority': 'u=1, i',
            'referer': 'https://www.zeptonow.com/',
            'sec-ch-ua': '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'tenant': 'ZEPTO',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
            'x-csrf-secret': 'fMtyolCemc8',
            'x-timezone': 'a59e9e20e6250fc48d3962e1b54f86e74aeb5c274d23fc93acc0e69c3af6748b',
            'x-without-bearer': 'true',
            'x-xsrf-token': 'R4KRG8Y1BoeZppJ-kK_f7:KiZEGh-nT-diexELryzJlRROmxY.VNNIOcTigOPaMpDZ3aHM2s+3SsniQ5tLgjCCJgeblKM'
        }
        
        # Generate dynamic headers
        self._generate_dynamic_headers()
    
    def _setup_session(self):
        """Set up session with realistic browser behavior"""
        # Add realistic cookies and session data
        self.session.cookies.update({
            '_gcl_au': '1.1.1546132427.1760714486',
            '_ga': 'GA1.1.2075143717.1760714486',
            '_fbp': 'fb.1.1760714488084.511467113732686430',
            '_ga_52LKG2B3L1': 'GS2.1.s1760714486$o1$g1$t1760714497$j49$l0$h634900868'
        })
        
        # Set realistic connection settings
        self.session.headers.update({
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        })
        
    def _generate_dynamic_headers(self):
        """Generate dynamic headers that change with each request"""
        # Generate unique IDs
        device_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Store IDs for consistency
        self.device_id = device_id
        self.request_id = request_id
        self.session_id = session_id
        
        # Add dynamic headers
        self.default_headers.update({
            'device_id': device_id,
            'deviceid': device_id,
            'request_id': request_id,
            'requestid': request_id,
            'session_id': session_id,
            'sessionid': session_id,
            'store_id': 'b4dc8d65-ed2e-4142-81b6-373982b13500',  # Default store ID
            'store_ids': 'b4dc8d65-ed2e-4142-81b6-373982b13500',
            'storeid': 'b4dc8d65-ed2e-4142-81b6-373982b13500',
            'store_etas': '{"b4dc8d65-ed2e-4142-81b6-373982b13500":-1}'
        })
        
        # Generate request signature (simplified version)
        self._generate_request_signature()
    
    def _generate_request_signature(self):
        """Generate request signature for authentication"""
        # This is a simplified version - in reality, Zepto might use more complex signing
        timestamp = str(int(time.time()))
        signature_data = f"{self.device_id}{timestamp}"
        signature = hashlib.sha256(signature_data.encode()).hexdigest()
        
        self.default_headers['request-signature'] = signature
    
    def search_products(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for products on Zepto using web scraping approach
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of product dictionaries
        """
        try:
            print(f"🔍 Zepto: Searching for '{query}'...")
            # Try API approach first
            api_results = self._search_via_api(query, max_results)
            if api_results:
                print(f"✅ Zepto: Found {len(api_results)} products via API")
                return api_results
            
            # Fallback to web scraping approach
            print(f"🔄 Zepto: API failed, trying web scraping for: {query}")
            web_results = self._search_via_web(query, max_results)
            if web_results:
                print(f"✅ Zepto: Found {len(web_results)} products via web scraping")
                return web_results
            
            # Final fallback to alternative approach
            print(f"🔄 Zepto: Web scraping failed, using alternative approach for: {query}")
            alt_results = self._try_alternative_approach(query, max_results)
            if alt_results:
                print(f"✅ Zepto: Found {len(alt_results)} products via alternative approach")
                return alt_results
            else:
                print(f"❌ Zepto: No products found for '{query}'")
                return []
                
        except Exception as e:
            print(f"❌ Zepto: Error searching '{query}': {str(e)}")
            return []
    
    def _search_via_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Try to search via API with multiple endpoints"""
        try:
            # Prepare search payload
            search_payload = {
                "query": query,
                "page": 1,
                "limit": max_results,
                "sort": "relevance",
                "filters": {},
                "store_id": "b4dc8d65-ed2e-4142-81b6-373982b13500"
            }
            
            # Try multiple endpoints
            endpoints_to_try = [self.search_endpoint] + self.alternative_endpoints
            
            for endpoint in endpoints_to_try:
                try:
                    print(f"Trying Zepto endpoint: {endpoint}")
                    
                    # Update headers for this request
                    headers = self.default_headers.copy()
                    headers.update({
                        'Referer': 'https://www.zeptonow.com/',
                        'Origin': 'https://www.zeptonow.com',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'
                    })
                    
                    # Try GET request first
                    get_url = f"{endpoint}?q={query.replace(' ', '+')}&limit={max_results}"
                    response = self.session.get(get_url, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = self._parse_search_results(data)
                        if results:
                            print(f"✅ Success with GET request to {endpoint}")
                            return results
                    
                    # Try POST request
                    response = self.session.post(
                        endpoint,
                        headers=headers,
                        json=search_payload,
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = self._parse_search_results(data)
                        if results:
                            print(f"✅ Success with POST request to {endpoint}")
                            return results
                        else:
                            print(f"❌ No results from {endpoint}")
                    else:
                        print(f"❌ API error {response.status_code} from {endpoint}")
                        
                except Exception as e:
                    print(f"❌ Error with {endpoint}: {str(e)}")
                    continue
            
            print("All API endpoints failed")
            return []
                
        except Exception as e:
            print(f"API search failed: {str(e)}")
            return []
    
    def _search_via_web(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search via web scraping as fallback"""
        try:
            # Use the main website search
            search_url = f"https://www.zeptonow.com/search?q={query.replace(' ', '+')}"
            
            # Update headers for web request
            web_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            
            # Removed delay for faster processing
            
            response = self.session.get(search_url, headers=web_headers, timeout=15)
            
            if response.status_code == 200:
                # Try to extract products from HTML
                return self._extract_products_from_html(response.text, query, max_results)
            else:
                print(f"Web search failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Web search error: {str(e)}")
            return []
    
    
    def _try_alternative_approach(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Try alternative approaches to get real data"""
        try:
            print(f"Trying alternative approach for: {query}")
            
            # Try direct web scraping with different headers
            search_url = f"https://www.zeptonow.com/search?q={query.replace(' ', '+')}"
            
            # Use more realistic headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0',
                'Referer': 'https://www.zeptonow.com/'
            }
            
            response = self.session.get(search_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Try to extract product data from HTML
                return self._extract_products_from_html(response.text, query, max_results)
            else:
                print(f"Web scraping failed: {response.status_code}")
                return []
            
        except Exception as e:
            print(f"Alternative approach failed: {str(e)}")
            return []
    
    def _extract_products_from_html(self, html_content: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Extract product data from HTML content"""
        try:
            import re
            import json
            
            # Method 1: Look for JSON data in script tags with various patterns
            json_patterns = [
                r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
                r'window\.__PRELOADED_STATE__\s*=\s*({.*?});',
                r'window\.__NEXT_DATA__\s*=\s*({.*?});',
                r'__NEXT_DATA__"\s*type="application/json">({.*?})</script>',
                r'window\.__APOLLO_STATE__\s*=\s*({.*?});'
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, html_content, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        # Try different data structures
                        products = self._extract_products_from_json(data)
                        if products:
                            return self._parse_products_from_data(products, max_results)
                    except json.JSONDecodeError:
                        continue
            
            # Method 2: Look for product data in script tags
            script_patterns = [
                r'<script[^>]*>.*?products.*?(\[.*?\])',
                r'<script[^>]*>.*?items.*?(\[.*?\])',
                r'<script[^>]*>.*?results.*?(\[.*?\])'
            ]
            
            for pattern in script_patterns:
                matches = re.findall(pattern, html_content, re.DOTALL)
                for match in matches:
                    try:
                        products = json.loads(match)
                        if isinstance(products, list) and len(products) > 0:
                            return self._parse_products_from_data(products, max_results)
                    except json.JSONDecodeError:
                        continue
            
            # Method 3: Direct HTML parsing for product cards
            products = self._parse_html_product_cards(html_content, max_results)
            if products:
                return products
            
            print("Could not extract products from HTML")
            return []
            
        except Exception as e:
            print(f"HTML extraction failed: {str(e)}")
            return []
    
    def _extract_products_from_json(self, data: dict) -> List[Dict]:
        """Extract products from various JSON data structures"""
        try:
            # Try different possible paths in the JSON
            paths_to_try = [
                ['search', 'products'],
                ['products'],
                ['data', 'products'],
                ['results', 'products'],
                ['items'],
                ['data', 'items'],
                ['search', 'results'],
                ['results']
            ]
            
            for path in paths_to_try:
                current = data
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        current = None
                        break
                
                if isinstance(current, list) and len(current) > 0:
                    return current
            
            return []
        except Exception:
            return []
    
    def _parse_html_product_cards(self, html_content: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse product cards directly from HTML"""
        try:
            import re
            
            # Look for product card patterns
            product_patterns = [
                r'<div[^>]*class="[^"]*product[^"]*"[^>]*>.*?</div>',
                r'<div[^>]*class="[^"]*item[^"]*"[^>]*>.*?</div>',
                r'<div[^>]*class="[^"]*card[^"]*"[^>]*>.*?</div>'
            ]
            
            products = []
            for pattern in product_patterns:
                matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
                
                for match in matches[:max_results]:
                    # Extract basic product info
                    name_match = re.search(r'<[^>]*class="[^"]*name[^"]*"[^>]*>([^<]+)</', match, re.IGNORECASE)
                    price_match = re.search(r'<[^>]*class="[^"]*price[^"]*"[^>]*>([^<]+)</', match, re.IGNORECASE)
                    
                    if name_match and price_match:
                        products.append({
                            'name': name_match.group(1).strip(),
                            'price': price_match.group(1).strip(),
                            'brand': '',
                            'rating': '',
                            'product_url': '',
                            'image_url': '',
                            'variant': '',
                            'inventory': 0,
                            'eta': ''
                        })
            
            return products[:max_results]
        except Exception:
            return []
    
    def _parse_products_from_data(self, products: List[Dict], max_results: int) -> List[Dict[str, Any]]:
        """Parse products from extracted data"""
        parsed_products = []
        
        for product in products[:max_results]:
            try:
                parsed_product = {
                    'name': product.get('name', ''),
                    'price': str(product.get('price', 0)),
                    'brand': product.get('brand', ''),
                    'variant': product.get('variant', ''),
                    'image_url': product.get('image_url', ''),
                    'product_url': product.get('url', ''),
                    'rating': product.get('rating', ''),
                    'review_count': product.get('review_count', ''),
                    'available': product.get('available', True),
                    'eta': product.get('eta', ''),
                    'platform': 'zepto'
                }
                parsed_products.append(parsed_product)
            except Exception as e:
                print(f"Error parsing product: {str(e)}")
                continue
        
        return parsed_products
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse search results from Zepto API response
        
        Args:
            data: Raw API response data
            
        Returns:
            List of parsed product dictionaries
        """
        products = []
        
        try:
            # Navigate through the response structure
            if 'data' in data and 'products' in data['data']:
                product_list = data['data']['products']
                
                for product in product_list:
                    parsed_product = self._parse_product(product)
                    if parsed_product:
                        products.append(parsed_product)
            
            # Alternative structure - check for different response formats
            elif 'products' in data:
                for product in data['products']:
                    parsed_product = self._parse_product(product)
                    if parsed_product:
                        products.append(parsed_product)
            
            # Another possible structure
            elif isinstance(data, list):
                for product in data:
                    parsed_product = self._parse_product(product)
                    if parsed_product:
                        products.append(parsed_product)
                        
        except Exception as e:
            print(f"Error parsing Zepto results: {str(e)}")
            print(f"Raw response structure: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        return products
    
    def _parse_product(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse individual product data
        
        Args:
            product: Raw product data from API
            
        Returns:
            Parsed product dictionary or None if parsing fails
        """
        try:
            # Extract basic product information
            name = product.get('name', '')
            price = product.get('price', 0)
            original_price = product.get('original_price', price)
            
            # Handle different price formats
            if isinstance(price, dict):
                price = price.get('value', 0)
            if isinstance(original_price, dict):
                original_price = original_price.get('value', price)
            
            # Extract rating
            rating = product.get('rating', 0)
            if isinstance(rating, dict):
                rating = rating.get('average', 0)
            
            # Extract variant/size information
            variant = product.get('variant', '')
            if not variant and 'size' in product:
                variant = product.get('size', '')
            
            # Extract brand
            brand = product.get('brand', '')
            if not brand and 'brand_name' in product:
                brand = product.get('brand_name', '')
            
            # Extract product URL
            product_url = product.get('url', '')
            if not product_url and 'product_url' in product:
                product_url = product.get('product_url', '')
            
            # Extract image URL
            image_url = product.get('image_url', '')
            if not image_url and 'images' in product:
                images = product.get('images', [])
                if images and len(images) > 0:
                    image_url = images[0].get('url', '') if isinstance(images[0], dict) else str(images[0])
            
            # Extract availability
            available = product.get('available', True)
            if isinstance(available, dict):
                available = available.get('in_stock', True)
            
            # Extract ETA
            eta = product.get('eta', '')
            if isinstance(eta, dict):
                eta = eta.get('text', '')
            
            # Create parsed product
            parsed_product = {
                'name': name,
                'price': str(price),
                'original_price': str(original_price),
                'rating': str(rating) if rating else '',
                'review_count': str(product.get('review_count', '')),
                'product_url': f"https://www.zeptonow.com{product_url}" if product_url and not product_url.startswith('http') else product_url,
                'image_url': image_url,
                'brand': brand,
                'variant': variant,
                'available': available,
                'eta': eta,
                'scraped_at': datetime.now().isoformat(),
                'platform': 'zepto'
            }
            
            return parsed_product
            
        except Exception as e:
            print(f"Error parsing individual product: {str(e)}")
            return None
    
    def get_product_details(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific product
        
        Args:
            product_id: Zepto product ID
            
        Returns:
            Detailed product information or None
        """
        try:
            # This would require a different endpoint for product details
            # For now, we'll return None as we don't have the details endpoint
            return None
            
        except Exception as e:
            print(f"Error getting product details: {str(e)}")
            return None
    
    def save_results(self, products: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Save search results to JSON file
        
        Args:
            products: List of product dictionaries
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zepto_data_{timestamp}.json"
        
        filepath = f"temp_output/{filename}"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(products, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved {len(products)} Zepto products to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return ""

def test_zepto_scraper():
    """Test function for Zepto scraper"""
    print("Testing Zepto Scraper...")
    
    scraper = ZeptoScraper()
    
    # Test search queries
    test_queries = [
        "milk",
        "bread",
        "eggs",
        "rice",
        "tomatoes"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: {query}")
        products = scraper.search_products(query, max_results=5)
        
        if products:
            print(f"Found {len(products)} products:")
            for i, product in enumerate(products[:3], 1):  # Show first 3
                print(f"  {i}. {product['name']} - ₹{product['price']}")
        else:
            print("  No products found")
        
        # Removed delay for faster processing
    
    # Save all results
    all_products = []
    for query in test_queries:
        products = scraper.search_products(query, max_results=10)
        all_products.extend(products)
        # Removed delay for faster processing
    
    if all_products:
        scraper.save_results(all_products)
        print(f"\n✅ Total products found: {len(all_products)}")

if __name__ == "__main__":
    test_zepto_scraper()
