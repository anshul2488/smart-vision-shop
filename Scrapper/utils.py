"""
Utility functions for Amazon scraper
"""

import urllib.parse
import re
from typing import List, Dict, Any


def build_amazon_url(item_name: str) -> str:
    """
    Build Amazon India search URL for a given item name.
    
    Args:
        item_name (str): The item to search for
        
    Returns:
        str: Complete Amazon India search URL
    """
    # URL encode the item name
    encoded_item = urllib.parse.quote_plus(item_name)
    
    # Build the Amazon India search URL with the specified format
    base_url = "https://www.amazon.in/s"
    params = {
        'k': item_name,
        'i': 'nowstore',
        'rh': 'n%3A16392737031'
    }
    
    # Create the complete URL
    url = f"{base_url}?k={encoded_item}&i=nowstore&rh=n%3A16392737031"
    
    return url


def build_blinkit_url(item_name: str) -> str:
    """
    Build Blinkit API URL for a given item name.
    
    Args:
        item_name (str): The item to search for
        
    Returns:
        str: Complete Blinkit API URL
    """
    # URL encode the item name
    encoded_item = urllib.parse.quote_plus(item_name)
    
    # Use the real Blinkit API endpoint discovered from browser network requests
    base_url = "https://blinkit.com/v1/layout/search"
    
    # Build the query parameters based on the real API call
    params = {
        'offset': '0',
        'limit': '50',
        'page_index': '1',
        'q': encoded_item,
        'search_count': '0',
        'search_method': 'basic',
        'search_type': 'type_to_search',
        'total_entities_processed': '0',
        'total_pagination_items': '0'
    }
    
    # Create the complete URL
    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"
    
    return url


def build_blinkit_alternative_urls(item_name: str) -> list:
    """
    Build alternative Blinkit API URLs to try different endpoints.
    
    Args:
        item_name (str): The item to search for
        
    Returns:
        list: List of possible API URLs
    """
    encoded_item = urllib.parse.quote_plus(item_name)
    
    # List of possible API endpoints to try
    possible_urls = [
        f"https://blinkit.com/api/v1/search?q={encoded_item}",
        f"https://blinkit.com/api/v1/products/search?query={encoded_item}",
        f"https://blinkit.com/api/search?q={encoded_item}",
        f"https://blinkit.com/api/v2/search?q={encoded_item}",
        f"https://blinkit.com/api/v1/catalog/search?q={encoded_item}",
        f"https://blinkit.com/api/v1/items/search?q={encoded_item}",
        f"https://blinkit.com/api/v1/groceries/search?q={encoded_item}",
        f"https://blinkit.com/api/v1/search/products?q={encoded_item}",
        f"https://blinkit.com/api/v1/search?query={encoded_item}",
        f"https://blinkit.com/api/v1/search?search={encoded_item}",
    ]
    
    return possible_urls


def build_blinkit_url_with_pagination(item_name: str, offset: int = 0, limit: int = 12) -> str:
    """
    Build Blinkit API URL with pagination parameters (based on current headers).
    
    Args:
        item_name (str): The item to search for
        offset (int): Offset for pagination
        limit (int): Number of items per page
        
    Returns:
        str: Complete Blinkit API URL with pagination
    """
    encoded_item = urllib.parse.quote_plus(item_name)
    
    # Build URL with pagination parameters from current headers
    base_url = "https://blinkit.com/v1/layout/search"
    
    params = {
        'offset': str(offset),
        'limit': str(limit),
        'last_snippet_type': 'product_card_snippet_type_2',
        'last_widget_type': 'listing_container',
        'page_index': '1',
        'q': encoded_item,
        'search_count': '21',  # Updated from current headers
        'search_method': 'basic',
        'search_type': 'type_to_search',
        'total_entities_processed': '1',
        'total_pagination_items': '21'
    }
    
    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"
    
    return url


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-.,₹$%()]', '', text)
    
    return text


def extract_price(price_text: str) -> str:
    """
    Extract and clean price information.
    
    Args:
        price_text (str): Raw price text
        
    Returns:
        str: Cleaned price
    """
    if not price_text:
        return ""
    
    # Remove extra whitespace
    price = clean_text(price_text)
    
    # Extract price pattern (₹ followed by numbers and commas)
    price_match = re.search(r'₹[\d,]+(?:\.\d{2})?', price)
    if price_match:
        return price_match.group()
    
    return price


def extract_rating(rating_text: str) -> str:
    """
    Extract rating from text.
    
    Args:
        rating_text (str): Raw rating text
        
    Returns:
        str: Cleaned rating
    """
    if not rating_text:
        return ""
    
    # Look for rating pattern (e.g., "4.5 out of 5 stars")
    rating_match = re.search(r'(\d+\.?\d*)\s*out\s*of\s*5', rating_text)
    if rating_match:
        return rating_match.group(1)
    
    # Look for simple decimal rating
    rating_match = re.search(r'(\d+\.?\d*)', rating_text)
    if rating_match:
        return rating_match.group(1)
    
    return clean_text(rating_text)


def extract_review_count(review_text: str) -> str:
    """
    Extract number of reviews from text.
    
    Args:
        review_text (str): Raw review count text
        
    Returns:
        str: Cleaned review count
    """
    if not review_text:
        return ""
    
    # Look for review count pattern
    review_match = re.search(r'([\d,]+)', review_text)
    if review_match:
        return review_match.group(1)
    
    return clean_text(review_text)


def validate_url(url: str) -> bool:
    """
    Validate if URL is a proper Amazon product URL.
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if valid Amazon URL
    """
    if not url:
        return False
    
    # Check if it's an Amazon URL
    return 'amazon.in' in url.lower() and '/dp/' in url.lower()


def format_output_data(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format and clean the scraped product data.
    
    Args:
        products (List[Dict[str, Any]]): Raw product data
        
    Returns:
        List[Dict[str, Any]]: Formatted product data
    """
    formatted_products = []
    
    for product in products:
        # Check if this is Blinkit data (has additional fields)
        is_blinkit = 'variant' in product or 'brand' in product or 'product_id' in product
        
        if is_blinkit:
            # Format Blinkit-specific data
            formatted_product = {
                'name': clean_text(product.get('name', '')),
                'price': extract_price(product.get('price', '')),
                'variant': clean_text(product.get('variant', '')),
                'brand': clean_text(product.get('brand', '')),
                'rating': extract_rating(product.get('rating', '')),
                'review_count': extract_review_count(product.get('review_count', '')),
                'product_url': product.get('product_url', ''),
                'image_url': product.get('image_url', ''),
                'product_id': product.get('product_id', ''),
                'inventory': product.get('inventory', 0),
                'merchant_id': product.get('merchant_id', ''),
                'merchant_type': product.get('merchant_type', ''),
                'eta': clean_text(product.get('eta', '')),
                'scraped_at': product.get('scraped_at', '')
            }
        else:
            # Format Amazon data (original format)
            formatted_product = {
                'name': clean_text(product.get('name', '')),
                'price': extract_price(product.get('price', '')),
                'rating': extract_rating(product.get('rating', '')),
                'review_count': extract_review_count(product.get('review_count', '')),
                'product_url': product.get('product_url', ''),
                'image_url': product.get('image_url', ''),
                'scraped_at': product.get('scraped_at', '')
            }
        
        # Only add products that have at least a name
        if formatted_product['name']:
            formatted_products.append(formatted_product)
    
    return formatted_products
