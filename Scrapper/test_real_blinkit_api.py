#!/usr/bin/env python3
"""
Test script for the real Blinkit API endpoint
"""

import requests
import json
import urllib.parse


def test_real_blinkit_api():
    """Test the real Blinkit API endpoint with correct headers."""
    print("=" * 60)
    print("🧪 Testing Real Blinkit API")
    print("=" * 60)
    
    test_item = "curd"
    print(f"Testing search for: {test_item}")
    
    # Build the correct API URL
    encoded_item = urllib.parse.quote_plus(test_item)
    url = f"https://blinkit.com/v1/layout/search?offset=0&limit=50&page_index=1&q={encoded_item}&search_count=0&search_method=basic&search_type=type_to_search&total_entities_processed=0&total_pagination_items=0"
    
    print(f"API URL: {url}")
    print()
    
    # Headers from the real browser request
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
    
    try:
        print("📊 Making POST request to Blinkit API...")
        response = requests.post(url, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print()
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✅ SUCCESS! Got JSON response")
                print(f"Response keys: {list(data.keys())}")
                print()
                
                # Check if it has the expected structure
                if 'response' in data:
                    response_data = data['response']
                    print(f"Response data keys: {list(response_data.keys())}")
                    
                    if 'snippets' in response_data:
                        snippets = response_data['snippets']
                        print(f"Found {len(snippets)} product snippets")
                        
                        if snippets:
                            print("\n📋 Sample product data:")
                            sample_snippet = snippets[0]
                            if 'data' in sample_snippet:
                                product_data = sample_snippet['data']
                                print(f"  Product name: {product_data.get('name', {}).get('text', 'N/A')}")
                                print(f"  Price: {product_data.get('normal_price', {}).get('text', 'N/A')}")
                                print(f"  Brand: {product_data.get('brand_name', {}).get('text', 'N/A')}")
                                print(f"  Variant: {product_data.get('variant', {}).get('text', 'N/A')}")
                                print(f"  Product ID: {product_data.get('identity', {}).get('id', 'N/A')}")
                                print(f"  Inventory: {product_data.get('inventory', 'N/A')}")
                                print(f"  Image URL: {product_data.get('image', {}).get('url', 'N/A')}")
                                
                                # Show rating if available
                                rating_data = product_data.get('rating', {})
                                if rating_data and rating_data.get('type') == 'bar':
                                    bar_data = rating_data.get('bar', {})
                                    rating_value = bar_data.get('value', 'N/A')
                                    review_count = bar_data.get('title', {}).get('text', 'N/A')
                                    print(f"  Rating: {rating_value}")
                                    print(f"  Reviews: {review_count}")
                        
                        print(f"\n🎉 API is working! Found {len(snippets)} products")
                        return True
                    else:
                        print("⚠️  No 'snippets' found in response")
                        print(f"Available keys: {list(response_data.keys())}")
                else:
                    print("⚠️  No 'response' key found in data")
                    print(f"Available keys: {list(data.keys())}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"Response text (first 500 chars): {response.text[:500]}")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response text: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False


if __name__ == "__main__":
    success = test_real_blinkit_api()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Blinkit API test completed successfully!")
        print("The scraper should now work with the correct endpoint.")
    else:
        print("❌ Blinkit API test failed")
        print("You may need to update the headers or check the API endpoint.")
    print("=" * 60)
