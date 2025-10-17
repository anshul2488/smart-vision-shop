#!/usr/bin/env python3
"""
Script to help find the correct Blinkit API endpoint
"""

import requests
import json
from utils import build_blinkit_alternative_urls


def test_blinkit_endpoints():
    """Test various Blinkit API endpoints to find the working one."""
    print("=" * 60)
    print("🔍 Finding Blinkit API Endpoint")
    print("=" * 60)
    
    test_item = "curd"
    print(f"Testing search for: {test_item}")
    
    # Get all possible URLs
    urls = build_blinkit_alternative_urls(test_item)
    
    # Add the correct API endpoint first
    correct_url = f"https://blinkit.com/v1/layout/search?offset=0&limit=50&page_index=1&q={test_item}&search_count=0&search_method=basic&search_type=type_to_search&total_entities_processed=0&total_pagination_items=0"
    
    # Add some additional common API patterns
    additional_urls = [
        correct_url,  # Add the correct endpoint first
        f"https://blinkit.com/api/v1/search?term={test_item}",
        f"https://blinkit.com/api/v1/search?keyword={test_item}",
        f"https://blinkit.com/api/v1/search?text={test_item}",
        f"https://blinkit.com/api/v1/search?item={test_item}",
        f"https://blinkit.com/api/v1/search?product={test_item}",
        f"https://blinkit.com/api/v1/search?name={test_item}",
        f"https://blinkit.com/api/v1/search?query={test_item}&limit=20",
        f"https://blinkit.com/api/v1/search?q={test_item}&limit=20",
        f"https://blinkit.com/api/v1/search?q={test_item}&page=1&limit=20",
    ]
    
    all_urls = urls + additional_urls
    
    print(f"Testing {len(all_urls)} different API endpoints...")
    print()
    
    # Headers to mimic a browser request (based on real Blinkit API call)
    headers = {
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
    }
    
    working_endpoints = []
    
    for i, url in enumerate(all_urls, 1):
        print(f"{i:2d}. Testing: {url}")
        
        try:
            # Use POST for the correct Blinkit API endpoint
            if 'v1/layout/search' in url:
                response = requests.post(url, headers=headers, timeout=10)
            else:
                response = requests.get(url, headers=headers, timeout=10)
            status_code = response.status_code
            
            if status_code == 200:
                try:
                    data = response.json()
                    print(f"    ✅ SUCCESS! Status: {status_code}")
                    print(f"    📊 Response keys: {list(data.keys())}")
                    
                    # Check if it looks like a product search response
                    if 'response' in data or 'snippets' in data or 'products' in data or 'items' in data:
                        print(f"    🎯 This looks like a product search API!")
                        working_endpoints.append({
                            'url': url,
                            'status': status_code,
                            'response_keys': list(data.keys()),
                            'sample_data': data
                        })
                    
                    # Show a small sample of the response
                    if isinstance(data, dict) and len(str(data)) < 500:
                        print(f"    📋 Sample response: {data}")
                    elif isinstance(data, dict):
                        print(f"    📋 Response size: {len(str(data))} characters")
                    
                except json.JSONDecodeError:
                    print(f"    ⚠️  Status: {status_code} (Not JSON)")
                    if len(response.text) < 200:
                        print(f"    📋 Response: {response.text}")
                except Exception as e:
                    print(f"    ❌ Error parsing response: {e}")
            
            elif status_code == 404:
                print(f"    ❌ Not Found (404)")
            elif status_code == 403:
                print(f"    🚫 Forbidden (403)")
            elif status_code == 401:
                print(f"    🔐 Unauthorized (401)")
            else:
                print(f"    ⚠️  Status: {status_code}")
                
        except requests.exceptions.Timeout:
            print(f"    ⏰ Timeout")
        except requests.exceptions.ConnectionError:
            print(f"    🔌 Connection Error")
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if working_endpoints:
        print(f"✅ Found {len(working_endpoints)} working endpoint(s):")
        for i, endpoint in enumerate(working_endpoints, 1):
            print(f"\n{i}. {endpoint['url']}")
            print(f"   Status: {endpoint['status']}")
            print(f"   Response keys: {endpoint['response_keys']}")
    else:
        print("❌ No working API endpoints found")
        print("\n💡 Suggestions:")
        print("1. Check Blinkit's website in browser and inspect network requests")
        print("2. Look for XHR/Fetch requests when searching for products")
        print("3. The API might require authentication or specific headers")
        print("4. The API might use a different domain or subdomain")
        print("5. Try using browser developer tools to find the actual API calls")
    
    return working_endpoints


if __name__ == "__main__":
    test_blinkit_endpoints()
