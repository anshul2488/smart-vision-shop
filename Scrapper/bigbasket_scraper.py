import requests
import json
import time
import random
import re
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
            # print(f"BigBasket: Searching for '{query}'...")  # Removed INFO log
            
            # Try API first with the working term completion endpoint
            results = self._try_api_search(query, max_results)
            if results:
                # print(f"BigBasket: Found {len(results)} products via API")  # Removed INFO log
                return results
            
            # Try web scraping as fallback
            # print("BigBasket API failed, trying web scraping...")  # Removed INFO log
            results = self._try_web_scraping(query, max_results)
            if results:
                # print(f"BigBasket: Found {len(results)} products via web scraping")  # Removed INFO log
                return results
            
            # print(f"BigBasket: No products found for '{query}'")  # Removed INFO log
            return []
            
        except Exception as e:
            print(f"BigBasket: Error searching '{query}': {e}")
            return []
    
    def _try_api_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Try BigBasket API search using the correct Next.js data endpoint"""
        try:
            # Use the working Next.js data API endpoint
            results = self._try_nextjs_data_api(query, max_results)
            if results:
                return results
            
            # Fallback to term completion API
            results = self._try_term_completion_api(query, max_results)
            if results:
                return results
            
            return []
                
        except Exception as e:
            print(f"Error in API search: {e}")
            return []
    
    def _try_nextjs_data_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Try BigBasket Next.js data API - the main working endpoint"""
        try:
            # Use the exact URL pattern from your network request
            api_url = f"https://www.bigbasket.com/_next/data/aIW3tY6XTmjLFEHfgaLjO/ps.json"
            
            # Use the exact headers from your working request
            headers = {
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
                'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
                'Cache-Control': 'no-cache',
                'Cookie': '_bb_locSrc=default; x-channel=web; _bb_aid=MjkxMzA4NDUzMA==; _bb_cid=1; _bb_vid=OTM3NDcyODUxOTM5NTQ5OTM0; _bb_nhid=7427; _bb_dsid=7427; _bb_dsevid=7427; _bb_bhid=; _bb_loid=; csrftoken=tH0LrCz91k8vOzHrUc7zNCLTZM6HXu7uTPbtizfE4TLb3KFgycDyEivzGBJiveZ7; isintegratedsa=true; _bb_bb2.0=1; is_global=1; _bb_addressinfo=; _bb_pin_code=; _is_bb1.0_supported=0; is_integrated_sa=1; bb2_enabled=true; ufi=1; bigbasket.com=345fd6f8-c339-4bca-b82e-dece5b8ef5b6; _gcl_au=1.1.2095448383.1759944938; _fbp=fb.1.1759944938571.148489199130814633; jentrycontextid=10; xentrycontextid=10; xentrycontext=bbnow; jarvis-id=313140c7-0440-44b3-a078-4c1c036aac9e; _bb_sa_ids=19224; _is_tobacco_enabled=1; _bb_cda_sa_info=djIuY2RhX3NhLjEwLjE5MjI0; adb=0; _gcl_aw=GCL.1761696376.CjwKCAjw04HIBhB8EiwA8jGNbTyOJB47faOj8IA6qiZTE0m5DLyIq9C5o7kwKqduQMQ5dfJS2Fq69xoCdkkQAvD_BwE; _gcl_gs=2.1.k1$i1761696373$u160754478; _gid=GA1.2.819797202.1761696376; _gac_UA-27455376-1=1.1761696376.CjwKCAjw04HIBhB8EiwA8jGNbTyOJB47faOj8IA6qiZTE0m5DLyIq9C5o7kwKqduQMQ5dfJS2Fq69xoCdkkQAvD_BwE; csurftoken=CVHjQA.OTM3NDcyODUxOTM5NTQ5OTM0.1761703121122.AvqI5IO1uvnSQRhyQbISPZsul9EmXzOjZHXW/HSkFQE=; ts=2025-10-29%2007:28:43.562; _gat_UA-27455376-1=1; _ga=GA1.1.2062664848.1759944938; _ga_FRRYG5VKHX=GS2.1.s1761703124$o9$g0$t1761703124$j60$l0$h0',
                'Pragma': 'no-cache',
                'Priority': 'u=1, i',
                'Referer': f'https://www.bigbasket.com/ps/?q={query}&nc=as',
                'Sec-Ch-Ua': '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
                'X-Nextjs-Data': '1'
            }
            
            params = {
                'q': query,
                'nc': 'as',
                'listing': 'ps'
            }
            
            # print(f"Trying Next.js data API: {api_url}")  # Removed INFO log
            
            response = requests.get(api_url, params=params, headers=headers, timeout=15)
            
            # print(f"Next.js data API status: {response.status_code}")  # Removed INFO log
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    # print(f"Next.js data API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")  # Removed INFO log
                    products = self._parse_nextjs_data_response(data, max_results)
                    if products:
                        return products
                except json.JSONDecodeError as e:
                    print(f"Next.js data API returned invalid JSON: {e}")
                    print(f"Response text: {response.text[:200]}")
            else:
                print(f"Next.js data API returned status code: {response.status_code}")
                print(f"Response text: {response.text[:200]}")
            
            return []
                
        except Exception as e:
            print(f"Error calling Next.js data API: {e}")
            return []
    
    def _parse_nextjs_data_response(self, data: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Parse Next.js data API response"""
        products = []
        
        try:
            # Navigate the response structure based on your actual response
            if 'pageProps' in data and 'SSRData' in data['pageProps']:
                ssr_data = data['pageProps']['SSRData']
                if 'tabs' in ssr_data and len(ssr_data['tabs']) > 0:
                    # Extract products from the first tab
                    first_tab = ssr_data['tabs'][0]
                    if 'product_info' in first_tab and 'products' in first_tab['product_info']:
                        product_list = first_tab['product_info']['products']
                        print(f"Found {len(product_list)} products in Next.js data response")
                        
                        for product in product_list[:max_results]:
                            parsed_product = self._extract_product_from_nextjs_data(product)
                            if parsed_product:
                                products.append(parsed_product)
                    else:
                        print("Could not find products in Next.js data response tabs")
                        return []
                else:
                    print("No tabs found in Next.js data response")
                    return []
            else:
                print("No pageProps.SSRData found in Next.js data response")
                return []
                
        except Exception as e:
            print(f"Error parsing Next.js data response: {e}")
        
        return products
    
    def _extract_product_from_nextjs_data(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product information from Next.js data API product data"""
        try:
            # Extract basic product information
            name = product.get('desc', '')
            brand = ''
            if 'brand' in product and isinstance(product['brand'], dict):
                brand = product['brand'].get('name', '')
            
            # Extract pricing information
            price = ''
            original_price = ''
            if 'pricing' in product and 'discount' in product['pricing']:
                discount_info = product['pricing']['discount']
                price = discount_info.get('prim_price', {}).get('sp', '')
                original_price = discount_info.get('mrp', '')
            
            # Extract availability information
            eta = ''
            if 'availability' in product:
                eta = product['availability'].get('short_eta', '')
            
            # Extract image URL
            image_url = ''
            if 'images' in product and len(product['images']) > 0:
                image_url = product['images'][0].get('m', '')
            
            # Extract product URL
            product_url = product.get('absolute_url', '')
            if product_url and not product_url.startswith('http'):
                product_url = 'https://www.bigbasket.com' + product_url
            
            # Extract variant information
            variant = product.get('w', '')
            
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
                    'variant': variant,
                    'image_url': image_url,
                    'product_url': product_url,
                    'rating': '',
                    'review_count': '',
                    'available': True,
                    'eta': eta,
                    'discount': discount,
                    'scraped_at': datetime.now().isoformat(),
                    'platform': 'bigbasket'
                }
                
        except Exception as e:
            print(f"Error extracting product from Next.js data: {e}")
        
        return None
    
    def _try_term_completion_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Try BigBasket term completion API - this one works!"""
        try:
            api_url = "https://www.bigbasket.com/listing-svc/v1/product/term-completion"
            
            # Use the exact headers from your working request
            headers = {
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
                'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
                'Cache-Control': 'no-cache',
                'Common-Client-Static-Version': '101',
                'Content-Type': 'application/json',
                'Cookie': '_bb_locSrc=default; x-channel=web; _bb_aid=MjkxMzA4NDUzMA==; _bb_cid=1; _bb_vid=OTM3NDcyODUxOTM5NTQ5OTM0; _bb_nhid=7427; _bb_dsid=7427; _bb_dsevid=7427; _bb_bhid=; _bb_loid=; csrftoken=tH0LrCz91k8vOzHrUc7zNCLTZM6HXu7uTPbtizfE4TLb3KFgycDyEivzGBJiveZ7; isintegratedsa=true; _bb_bb2.0=1; is_global=1; _bb_addressinfo=; _bb_pin_code=; _is_bb1.0_supported=0; is_integrated_sa=1; bb2_enabled=true; ufi=1; bigbasket.com=345fd6f8-c339-4bca-b82e-dece5b8ef5b6; _gcl_au=1.1.2095448383.1759944938; _fbp=fb.1.1759944938571.148489199130814633; jentrycontextid=10; xentrycontextid=10; xentrycontext=bbnow; jarvis-id=313140c7-0440-44b3-a078-4c1c036aac9e; _bb_sa_ids=19224; _is_tobacco_enabled=1; _bb_cda_sa_info=djIuY2RhX3NhLjEwLjE5MjI0; bm_ss=ab8e18ef4e; adb=0; _gcl_aw=GCL.1761696376.CjwKCAjw04HIBhB8EiwA8jGNbTyOJB47faOj8IA6qiZTE0m5DLyIq9C5o7kwKqduQMQ5dfJS2Fq69xoCdkkQAvD_BwE; _gcl_gs=2.1.k1$i1761696373$u160754478; _gid=GA1.2.819797202.1761696376; _gac_UA-27455376-1=1.1761696376.CjwKCAjw04HIBhB8EiwA8jGNbTyOJB47faOj8IA6qiZTE0m5DLyIq9C5o7kwKqduQMQ5dfJS2Fq69xoCdkkQAvD_BwE; _ga=GA1.1.2062664848.1759944938; bm_lso=60F1E5C868746A5A19F95275C94C8819D88BB5EE07A023123F65491C4219C552~YAAQJRzFF4f5ye2ZAQAAE11JLQURg8t0ouJWWwYTBSyI7wFPxMzaThgcl5XwiPwEV4hfGBvOKrEqAHhnMtKPBww2idB/hS3tbT3u6dk7gkvl7xPibws4ahJNsO/0G5z6dwz6Sx4XfsyuKQXSDHyoQidEcNlasO5HWSt4z18vFosAmbxf5T9pg7sXPQKTfLPzdxSDSMTFZk/KJDYP7cBRRU0a3CzCXmFv9zUgGylnd+4sxD/W/KxnE7EXWFwzCPNtWLnyBK7arCXUmX/GjkkSgL0KUzNgiz4RiXDzNM+Eg8tmf6entPJihrcyOdhWHtV5K8/mfo/nYcorA8NsLA09M0J0DmATTE1xw8kI2dl7ijbvkcgz3Wa6VnzVwBwUbw28Z+mOY2qiRYaF/ZoyPmL0qHuXV5FQWjI8vcHI0f/Dhki2cYdkVEFjRPgq2h5rk2NbGK/2P6BxLT6hr9C8toSR^1761696377172; ts=2025-10-29%2005:36:23.162; _ga_FRRYG5VKHX=GS2.1.s1761696376$o8$g1$t1761696383$j53$l0$h0; bm_s=YAAQJRzFFwxiyu2ZAQAAOCthLQQnZuBWSFcV3eDeEM3/R3yR2Wj1X0Sg7sJz928tYjGnuzvnrJUDlyXayaHBltsztpDwcKeBNg9R9uNSwHf5369jHF8fuIEUs9L0kI1wBsJYJ+nTDQIMq0pOgB0hd7bCk5qZ+7Z+/I0KsLSGIl03YoZ24itjRrqZwaIuKYXIiQlupXbah+bDAclszxY08YvAmRHBpxKHyQ1dNnaaEzgOZuRQCufuPweJZFPYNau4R8PISFU5FHGpXKvvjHru9IGM1QculXtAlTOxzCZ+ErXUgwn2BfCXEATg5j+fRhqEeUE0sfGt0SOkHOBfW/ImrwddjYstpuD1uckXNb8JjMp44SdVWwubH7UaZ7xs/ubXhd2UvIqrRuufuo0tTQRRpclIX3kAM9t8fbcQE2TDSuWkBE8Bq/0AtH9lW3mOApXI2H5Aqg8NIGjh6+JCU3o11pHX5C1t9gKksvzrnHS69GLM3/h+SFvsOZDrOtmRH3NXXoXzpZiWMrsTvC9m3iVGAxAp4JeOGLUvcR77FUNY5aOj6oOzWMuloqXPpvdryOCE4UyeQw==; bm_so=805BB831390DFA1A23D9687A86A5C71B67609A5F0D4A7A01D59D55E78C7F347B~YAAQJRzFFw1iyu2ZAQAAOCthLQU5dMYH5ELZjXMjxCRICFTz/bkUEtntUhMvbMykwbii+TKI6fAz9w9Y8V2ZbPFgdgW0DpT1agDclCGW/I5D6UjMYko4k6bzJpDOQUaGvQF08yGeBQ4uXz+b7hp9P9PrGUmUXjBb5/iF9Zig2yVCE/qvoRz+Up4ui/0AxZ4g7RpcqeWHiDv1p9Mxheao7HtJmM5zK1NLY/K4Ja9KZgRPRCcioQSgGjpPkjI0FeMln+UqncXmPWwkw1uKl6Wk6+H4YlDnMr2kGz4kd5HlWJAmYap29F6JBegQfCxqdCxSaPOSPpRdoNDdxS3S3gj/B/+HZegqa9BnAp8CiaSxa+VJPLuloK9mvaqlTy+dKKxaGpM2nhfUyLpZJxcqPhibSZResuC+NhJVgejgGBL1bzUY86EymATwi0mLA/8v7vpnq4HWm2CV7dRw988ntbnM; csurftoken=WNgPAA.OTM3NDcyODUxOTM5NTQ5OTM0.1761697936514.3EX0r75hxrXvLPT7RsLiQYTVB6vxkwjc5+O0BV4zNS8=',
                'Pragma': 'no-cache',
                'Priority': 'u=1, i',
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
                'X-Tracker': '24214b23-73e8-48b4-bb91-f3fece3bd588'
            }
            
            params = {
                'term': query
            }
            
            # print(f"Trying term completion API: {api_url}")  # Removed INFO log
            
            # Remove compression headers to get raw response
            headers_no_compression = headers.copy()
            headers_no_compression.pop('Accept-Encoding', None)
            
            response = requests.get(api_url, params=params, headers=headers_no_compression, timeout=15)
            
            # print(f"Term completion API status: {response.status_code}")  # Removed INFO log
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"Term completion API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    products = self._parse_term_completion_response(data, max_results)
                    if products:
                        return products
                except json.JSONDecodeError as e:
                    print(f"Term completion API returned invalid JSON: {e}")
                    print(f"Response text: {response.text[:200]}")
                    # Try to decode as text to see what we got
                    try:
                        print(f"Raw response content: {response.content[:200]}")
                    except:
                        pass
            elif response.status_code == 403:
                print("Term completion API blocked with 403 - anti-bot protection")
            else:
                print(f"Term completion API returned status code: {response.status_code}")
                print(f"Response text: {response.text[:200]}")
            
            return []
                
        except Exception as e:
            print(f"Error calling term completion API: {e}")
            return []
    
    def _parse_term_completion_response(self, data: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Parse term completion API response"""
        products = []
        
        try:
            # Extract products from the term completion response
            if 'products' in data and isinstance(data['products'], list):
                product_list = data['products']
                print(f"Found {len(product_list)} products in term completion response")
                
                for product in product_list[:max_results]:
                    parsed_product = self._extract_product_from_term_completion(product)
                    if parsed_product:
                        products.append(parsed_product)
            else:
                print("No products found in term completion response")
                return []
                
        except Exception as e:
            print(f"Error parsing term completion response: {e}")
        
        return products
    
    def _extract_product_from_term_completion(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product information from term completion API product data"""
        try:
            # Extract basic product information
            name = product.get('desc', '')
            brand = ''
            if 'brand' in product and isinstance(product['brand'], dict):
                brand = product['brand'].get('name', '')
            
            # Extract pricing information
            price = ''
            original_price = ''
            if 'pricing' in product and 'discount' in product['pricing']:
                discount_info = product['pricing']['discount']
                price = discount_info.get('prim_price', {}).get('sp', '')
                original_price = discount_info.get('mrp', '')
            
            # Extract availability information
            eta = ''
            if 'availability' in product:
                eta = product['availability'].get('short_eta', '')
            
            # Extract image URL
            image_url = ''
            if 'images' in product and len(product['images']) > 0:
                image_url = product['images'][0].get('s', '')
            
            # Extract product URL
            product_url = product.get('absolute_url', '')
            if product_url and not product_url.startswith('http'):
                product_url = 'https://www.bigbasket.com' + product_url
            
            # Extract variant information
            variant = product.get('w', '')
            
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
                    'variant': variant,
                    'image_url': image_url,
                    'product_url': product_url,
                    'rating': '',
                    'review_count': '',
                    'available': True,
                    'eta': eta,
                    'discount': discount,
                    'scraped_at': datetime.now().isoformat(),
                    'platform': 'bigbasket'
                }
                
        except Exception as e:
            print(f"Error extracting product from term completion: {e}")
        
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
    
    def _parse_api_response(self, data: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Parse API response data using the exact structure from network tab"""
        products = []
        
        try:
            # Navigate the response structure based on the actual response from network tab
            if 'tabs' in data and len(data['tabs']) > 0:
                # Extract products from the first tab
                first_tab = data['tabs'][0]
                if 'product_info' in first_tab and 'products' in first_tab['product_info']:
                    product_list = first_tab['product_info']['products']
                    print(f"Found {len(product_list)} products in API response")
                    
                    for product in product_list[:max_results]:
                        parsed_product = self._extract_product_from_api(product)
                        if parsed_product:
                            products.append(parsed_product)
                else:
                    print("Could not find products in BigBasket API response tabs")
                    return []
            else:
                print("No tabs found in BigBasket API response")
                return []
                
        except Exception as e:
            print(f"Error parsing API response: {e}")
        
        return products
    
    def _extract_product_from_api(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product information from BigBasket API product data"""
        try:
            # Extract basic product information
            name = product.get('desc', '')
            brand = ''
            if 'brand' in product and isinstance(product['brand'], dict):
                brand = product['brand'].get('name', '')
            
            # Extract pricing information
            price = ''
            original_price = ''
            if 'pricing' in product and 'discount' in product['pricing']:
                discount_info = product['pricing']['discount']
                price = discount_info.get('prim_price', {}).get('sp', '')
                original_price = discount_info.get('mrp', '')
            
            # Extract availability information
            eta = ''
            if 'availability' in product:
                eta = product['availability'].get('short_eta', '')
            
            # Extract image URL
            image_url = ''
            if 'images' in product and len(product['images']) > 0:
                image_url = product['images'][0].get('m', '')
            
            # Extract product URL
            product_url = product.get('absolute_url', '')
            if product_url and not product_url.startswith('http'):
                product_url = 'https://www.bigbasket.com' + product_url
            
            # Extract variant information
            variant = product.get('w', '')
            
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
                    'variant': variant,
                    'image_url': image_url,
                    'product_url': product_url,
                    'rating': '',
                    'review_count': '',
                    'available': True,
                    'eta': eta,
                    'discount': discount,
                    'scraped_at': datetime.now().isoformat(),
                    'platform': 'bigbasket'
                }
                
        except Exception as e:
            print(f"Error extracting product from API: {e}")
        
        return None
    
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
    products = scraper.search_products('milk', 3)
    print(f"Found {len(products)} products")
    for p in products:
        print(f"- {p['name']}: {p['price']} by {p['brand']}")
