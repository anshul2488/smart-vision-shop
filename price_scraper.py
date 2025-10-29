"""
Price Scraper for Grocery Items
Integrates with Amazon and D-Mart scrapers to get prices
"""
import sys
import os
import json
from typing import Dict, List, Any
from datetime import datetime

# Add Scrapper directory to path to import scrapers
scrapper_path = os.path.join(os.path.dirname(__file__), 'Scrapper')
sys.path.append(scrapper_path)

try:
    from amazon_scraper import AmazonScraper
    from zepto_scraper_advanced import AdvancedZeptoScraper as ZeptoScraper
    from bigbasket_scraper import BigBasketScraper
    try:
        from jiomart_scraper import JioMartScraper
        JIOMART_AVAILABLE = True
    except ImportError:
        JIOMART_AVAILABLE = False
    # Note: dmart_scraper not available
    SCRAPERS_AVAILABLE = True
    # Available scrapers imported successfully
except ImportError as e:
    print(f"Scrapers not available: {e}")
    SCRAPERS_AVAILABLE = False
    JIOMART_AVAILABLE = False

class PriceScraper:
    """Scrape prices for grocery items from multiple platforms"""
    
    def __init__(self):
        self.scrapers = {}
        if SCRAPERS_AVAILABLE:
            self.scrapers['amazon'] = AmazonScraper()
            self.scrapers['zepto'] = ZeptoScraper()
            self.scrapers['bigbasket'] = BigBasketScraper()
            if JIOMART_AVAILABLE:
                self.scrapers['jiomart'] = JioMartScraper()
            # Initialized scrapers successfully
        else:
            print("Scrapers not available. Will use sample data.")
    
    def scrape_item_prices(self, item_name: str, max_results: int = 5, save_individual: bool = False) -> Dict[str, List[Dict]]:
        """
        Scrape prices for a single item from all platforms (optimized with concurrent processing)
        
        Args:
            item_name: Name of the item to scrape
            max_results: Maximum results per platform
            save_individual: Whether to save individual scraper results (default: False)
            
        Returns:
            Dictionary with platform results
        """
        results = {}
        
        # Use concurrent processing for faster scraping
        import concurrent.futures
        import threading
        
        def scrape_platform(platform_scraper):
            platform, scraper = platform_scraper
            try:
                if platform == 'zepto':
                    # Zepto uses search_products method
                    products = scraper.search_products(item_name, max_results)
                    
                    # Convert Zepto format to standard format
                    standardized_products = []
                    for product in products[:max_results]:
                        standardized_products.append({
                            'name': product.get('name', ''),
                            'price': self._extract_price_number(product.get('price', '')),
                            'price_text': product.get('price', ''),
                            'brand': product.get('brand', ''),
                            'rating': product.get('rating', ''),
                            'review_count': product.get('review_count', ''),
                            'product_url': product.get('product_url', ''),
                            'image_url': product.get('image_url', ''),
                            'variant': product.get('variant', ''),
                            'inventory': product.get('available', True),
                            'eta': product.get('eta', ''),
                            'total_products_found': len(products),
                            'match_score': 15  # Default match score for Zepto
                        })
                    return platform, standardized_products
                elif platform == 'bigbasket':
                    # BigBasket uses search_products method
                    products = scraper.search_products(item_name, max_results)
                    
                    # Convert BigBasket format to standard format
                    standardized_products = []
                    for product in products[:max_results]:
                        standardized_products.append({
                            'name': product.get('name', ''),
                            'price': self._extract_price_number(product.get('price', '')),
                            'price_text': product.get('price', ''),
                            'brand': product.get('brand', ''),
                            'rating': product.get('rating', ''),
                            'review_count': product.get('review_count', ''),
                            'product_url': product.get('product_url', ''),
                            'image_url': product.get('image_url', ''),
                            'variant': product.get('variant', ''),
                            'inventory': product.get('available', True),
                            'eta': product.get('eta', ''),
                            'total_products_found': len(products),
                            'match_score': 15  # Default match score for BigBasket
                        })
                    return platform, standardized_products
                elif platform == 'jiomart':
                    # JioMart uses search_products method
                    products = scraper.search_products(item_name, max_results)
                    
                    # Convert JioMart format to standard format
                    standardized_products = []
                    for product in products:
                        standardized_products.append({
                            'name': product.get('name', ''),
                            'price': self._extract_price_number(product.get('price', '')),
                            'price_text': product.get('price', ''),
                            'brand': product.get('brand', ''),
                            'rating': product.get('rating', ''),
                            'review_count': product.get('review_count', ''),
                            'product_url': product.get('product_url', ''),
                            'image_url': product.get('image_url', ''),
                            'variant': product.get('variant', ''),
                            'inventory': product.get('inventory', 0),
                            'eta': product.get('eta', ''),
                            'total_products_found': len(products),
                            'match_score': 12  # Default match score for JioMart
                        })
                    return platform, standardized_products
                else:
                    # Amazon and D-Mart use the standard method
                    if save_individual:
                        products = scraper.search_products(item_name, max_results)
                    else:
                        # Don't save individual results, just get the data
                        products = scraper._search_products_data_only(item_name, max_results)
                    
                    # Standardize Amazon products too
                    standardized_products = []
                    for product in products[:max_results]:
                        standardized_products.append({
                            'name': product.get('name', ''),
                            'price': self._extract_price_number(product.get('price', '')),
                            'price_text': product.get('price', ''),
                            'brand': product.get('brand', ''),
                            'rating': product.get('rating', ''),
                            'review_count': product.get('review_count', ''),
                            'product_url': product.get('product_url', ''),
                            'image_url': product.get('image_url', ''),
                            'variant': product.get('variant', ''),
                            'inventory': product.get('inventory', 0),
                            'eta': product.get('eta', '')
                        })
                    return platform, standardized_products
            except Exception as e:
                return platform, []
        
        # Execute scraping concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_platform = {
                executor.submit(scrape_platform, (platform, scraper)): platform 
                for platform, scraper in self.scrapers.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_platform):
                platform, products = future.result()
                results[platform] = products
        
        return results
    
    def _extract_price_number(self, price_text: str) -> float:
        """Extract numeric price from price text"""
        if not price_text:
            return 0.0
        
        import re
        # Remove currency symbols and extract numbers
        price_match = re.search(r'[\d,]+(?:\.\d{2})?', price_text.replace(',', ''))
        if price_match:
            try:
                return float(price_match.group())
            except ValueError:
                return 0.0
        return 0.0
    
    def _find_best_matching_product(self, products: List[Dict], user_item: Dict) -> Dict:
        """
        Find the best matching product based on user input
        
        Args:
            products: List of products from scraper
            user_item: User's item with name, quantity, unit
            
        Returns:
            Best matching product with calculated total price
        """
        if not products:
            return None
        
        user_name = user_item.get('item_name', '').lower()
        user_quantity = user_item.get('quantity', '1')
        user_unit = user_item.get('unit', '').lower()
        
        best_match = None
        best_score = 0
        
        # First, filter products that match the name
        matching_products = []
        for product in products:
            product_name = product.get('name', '').lower()
            product_variant = product.get('variant', '').lower()
            
            # Basic name matching
            name_match = user_name in product_name or user_name == product_name
            if name_match:
                matching_products.append(product)
        
        # If no name matches, use all products
        if not matching_products:
            matching_products = products
        
        # Sort by rating first (highest first), then by price (lowest first)
        def sort_key(product):
            # Extract rating
            try:
                rating = float(product.get('rating', 0))
            except (ValueError, TypeError):
                rating = 0
            
            # Extract price and convert to float
            try:
                price = float(product.get('price', 0))
            except (ValueError, TypeError):
                price = 0
            
            # Return tuple for sorting: (-rating, price)
            # Negative rating for descending order, positive price for ascending order
            return (-rating, price)
        
        # Sort products by rating (highest first), then price (lowest first)
        matching_products.sort(key=sort_key)
        
        # Take the best product (first after sorting)
        best_match = matching_products[0] if matching_products else None
        best_score = 0
        
        # Calculate match score for display purposes
        if best_match:
            product_name = best_match.get('name', '').lower()
            product_variant = best_match.get('variant', '').lower()
            
            # Name matching score
            if user_name in product_name:
                best_score += 10
            if user_name == product_name:
                best_score += 5
            
            # Unit/variant matching
            if user_unit and user_unit in product_variant:
                best_score += 3
            
            # Brand matching
            if 'brand' in user_item and user_item['brand'].lower() in product_name:
                best_score += 2
            
            # Rating bonus
            try:
                rating = float(best_match.get('rating', 0))
                if rating > 4.0:
                    best_score += 2
                elif rating > 3.0:
                    best_score += 1
            except (ValueError, TypeError):
                pass
        
        # If no good match found, return the first product
        if not best_match and products:
            best_match = products[0]
        
        if best_match:
            # Calculate total price based on user quantity and product size
            try:
                raw_price = float(best_match.get('price', 0))
            except (ValueError, TypeError):
                raw_price = 0.0
            
            product_variant = best_match.get('variant', '').lower()
            
            try:
                user_qty = float(user_quantity)
                
                # Try to extract product size from variant
                product_size = self._extract_product_size(product_variant, user_unit)
                
                if product_size > 0:
                    # Calculate unit price (price per user unit)
                    unit_price = raw_price / product_size
                    # Calculate total price based on user quantity
                    total_price = unit_price * user_qty
                    units_needed = user_qty / product_size
                else:
                    # Fallback: assume 1 unit = 1 quantity (for items like "pieces")
                    # This handles cases where user wants "1 pieces" and product is sold as "1 piece"
                    unit_price = raw_price
                    total_price = unit_price * user_qty
                    units_needed = user_qty
                
                # Ensure calculated_total is never more than 5x the raw price (safety check)
                if total_price > raw_price * 5:
                    total_price = raw_price * user_qty
                    unit_price = raw_price
                
                best_match['calculated_total'] = total_price
                best_match['unit_price'] = unit_price  # Price per user unit
                best_match['user_quantity'] = user_quantity
                best_match['user_unit'] = user_unit
                best_match['match_score'] = best_score
                best_match['units_needed'] = units_needed
                best_match['product_size'] = product_size
                
            except (ValueError, TypeError):
                best_match['calculated_total'] = raw_price
                best_match['unit_price'] = raw_price
                best_match['user_quantity'] = user_quantity
                best_match['user_unit'] = user_unit
                best_match['match_score'] = best_score
                best_match['units_needed'] = 1
                best_match['product_size'] = 0
        
        return best_match
    
    def _extract_product_size(self, variant: str, user_unit: str) -> float:
        """
        Extract product size from variant string and convert to user unit
        
        Args:
            variant: Product variant string (e.g., "250ml", "1kg", "6 pieces")
            user_unit: User's requested unit (e.g., "L", "kg", "pieces")
            
        Returns:
            Product size in user's unit, or 0 if conversion not possible
        """
        import re
        
        if not variant or not user_unit:
            return 0
        
        # Extract number and unit from variant
        variant_match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', variant)
        if not variant_match:
            return 0
        
        variant_number = float(variant_match.group(1))
        variant_unit = variant_match.group(2).lower()
        user_unit_lower = user_unit.lower()
        
        # Unit conversion mappings
        conversions = {
            # Volume conversions (to liters)
            'ml': 0.001, 'milliliter': 0.001, 'millilitre': 0.001,
            'l': 1.0, 'liter': 1.0, 'litre': 1.0,
            'dl': 0.1, 'deciliter': 0.1, 'decilitre': 0.1,
            
            # Weight conversions (to kg)
            'g': 0.001, 'gram': 0.001, 'gm': 0.001,
            'kg': 1.0, 'kilogram': 1.0, 'kilo': 1.0,
            'mg': 0.000001, 'milligram': 0.000001,
            
            # Count conversions (to pieces)
            'piece': 1.0, 'pieces': 1.0, 'pcs': 1.0, 'pc': 1.0,
            'dozen': 12.0, 'dz': 12.0,
            'pack': 1.0, 'packs': 1.0, 'packet': 1.0, 'packets': 1.0,
        }
        
        # Convert variant to base unit
        variant_base = variant_number * conversions.get(variant_unit, 0)
        if variant_base == 0:
            return 0
        
        # Convert to user unit
        user_base = conversions.get(user_unit_lower, 0)
        if user_base == 0:
            # If user unit is not in conversions (like "pieces"), return 0
            # This will trigger the fallback logic in the calling function
            return 0
        
        # Return size in user's unit
        return variant_base / user_base
    
    def get_best_prices(self, user_item: Dict, max_results: int = 5) -> Dict[str, Any]:
        """
        Get the best prices for an item across all platforms with smart matching
        
        Args:
            user_item: Dictionary with item_name, quantity, unit
            max_results: Maximum results per platform
            
        Returns:
            Dictionary with best price information
        """
        item_name = user_item.get('item_name', '')
        
        # Scrape prices from all platforms
        all_results = self.scrape_item_prices(item_name, max_results)
        
        # Find best matching products for each platform
        best_prices = {
            'item': item_name,
            'user_quantity': user_item.get('quantity', '1'),
            'user_unit': user_item.get('unit', ''),
            'best_overall': None,
            'platforms': {},
            'all_products': all_results
        }
        
        for platform, products in all_results.items():
            if products:
                # Find best matching product on this platform
                best_match = self._find_best_matching_product(products, user_item)
                if best_match:
                    best_prices['platforms'][platform] = {
                        'best_price': best_match.get('price', 0),
                        'best_price_text': best_match.get('price_text', ''),
                        'product_name': best_match.get('name', ''),
                        'brand': best_match.get('brand', ''),
                        'variant': best_match.get('variant', ''),
                        'calculated_total': best_match.get('calculated_total', 0),
                        'match_score': best_match.get('match_score', 0),
                        'total_products': len(products),
                        'product_url': best_match.get('product_url', ''),
                        'image_url': best_match.get('image_url', ''),
                        'rating': best_match.get('rating', ''),
                        'review_count': best_match.get('review_count', ''),
                        'eta': best_match.get('eta', ''),
                        'inventory': best_match.get('inventory', 0)
                    }
        
        # Find overall best price based on calculated total
        all_best_prices = []
        for platform, info in best_prices['platforms'].items():
            if info['calculated_total'] > 0:
                all_best_prices.append({
                    'platform': platform,
                    'price': info['calculated_total'],
                    'unit_price': info['best_price'],
                    'product': info['product_name'],
                    'brand': info['brand'],
                    'variant': info['variant'],
                    'match_score': info['match_score']
                })
        
        if all_best_prices:
            # Sort by price first, then by match score
            def sort_key_best(x):
                try:
                    price = float(x['price'])
                except (ValueError, TypeError):
                    price = float('inf')  # Put invalid prices at the end
                
                try:
                    match_score = float(x['match_score'])
                except (ValueError, TypeError):
                    match_score = 0
                
                return (price, -match_score)
            
            best_overall = min(all_best_prices, key=sort_key_best)
            best_prices['best_overall'] = best_overall
        
        return best_prices
    
    def scrape_grocery_list_prices(self, grocery_dict: Dict[str, Any], save_to_files: bool = True) -> Dict[str, Any]:
        """
        Scrape prices for all items in a grocery list
        
        Args:
            grocery_dict: Dictionary with items and their details (name, quantity, unit)
            save_to_files: Whether to save all results to files at the end
            
        Returns:
            Dictionary with prices for all items
        """
        # Scraping prices for grocery list
        
        grocery_prices = {
            'timestamp': datetime.now().isoformat(),
            'total_items': len(grocery_dict),
            'items': {},
            'summary': {
                'total_estimated_cost': 0,
                'platforms_used': list(self.scrapers.keys()),
                'items_with_prices': 0
            },
            'all_products_found': {}  # Store all products found for each item
        }
        
        total_cost = 0
        items_with_prices = 0
        
        for item_key, item_data in grocery_dict.items():
            # Handle both old format (item: quantity) and new format (item: {quantity, unit})
            if isinstance(item_data, dict):
                item_name = item_data.get('item_name', item_key)
                quantity = item_data.get('quantity', 1)
                unit = item_data.get('unit', '')
            else:
                item_name = item_key
                quantity = item_data
                unit = ''
            
            # Processing item  # Removed INFO log
            
            # Create user item dictionary
            user_item = {
                'item_name': item_name,
                'quantity': str(quantity),
                'unit': unit
            }
            
            # Get best prices for this item (without saving individual files)
            item_prices = self.get_best_prices(user_item, max_results=5)
            
            # Store all products found for this item
            grocery_prices['all_products_found'][item_name] = item_prices['all_products']
            
            # Calculate cost for the quantity
            if item_prices['best_overall']:
                total_item_cost = item_prices['best_overall']['price']
                total_cost += total_item_cost
                items_with_prices += 1
                
                item_info = {
                    'quantity': quantity,
                    'unit': unit,
                    'unit_price': item_prices['best_overall']['unit_price'],
                    'total_cost': total_item_cost,
                    'best_platform': item_prices['best_overall']['platform'],
                    'best_product': item_prices['best_overall']['product'],
                    'best_brand': item_prices['best_overall']['brand'],
                    'best_variant': item_prices['best_overall']['variant'],
                    'match_score': item_prices['best_overall']['match_score'],
                    'all_prices': item_prices['platforms']
                }
            else:
                item_info = {
                    'quantity': quantity,
                    'unit': unit,
                    'unit_price': 0,
                    'total_cost': 0,
                    'best_platform': None,
                    'best_product': None,
                    'best_brand': None,
                    'best_variant': None,
                    'match_score': 0,
                    'all_prices': item_prices['platforms']
                }
            
            grocery_prices['items'][item_name] = item_info
        
        # Update summary
        grocery_prices['summary']['total_estimated_cost'] = total_cost
        grocery_prices['summary']['items_with_prices'] = items_with_prices
        
        # Save all results to files if requested
        if save_to_files:
            self._save_all_results(grocery_prices)
        
        return grocery_prices
    
    def _save_all_results(self, grocery_prices: Dict[str, Any]):
        """
        Save all accumulated results to JSON and CSV files
        
        Args:
            grocery_prices: Complete grocery prices data
        """
        try:
            import json
            import csv
            import os
            
            # Create output directory
            output_dir = "pipeline_results"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save complete grocery prices
            prices_file = os.path.join(output_dir, f"complete_results_{timestamp}.json")
            with open(prices_file, 'w', encoding='utf-8') as f:
                json.dump(grocery_prices, f, indent=2, ensure_ascii=False)
            
            # Save extracted items (best matches only)
            extracted_items = {}
            for item_name, item_info in grocery_prices['items'].items():
                if item_info['best_platform']:
                    extracted_items[item_name] = {
                        'quantity': item_info['quantity'],
                        'unit': item_info['unit'],
                        'best_platform': item_info['best_platform'],
                        'best_product': item_info['best_product'],
                        'best_brand': item_info['best_brand'],
                        'best_variant': item_info['best_variant'],
                        'unit_price': item_info['unit_price'],
                        'total_cost': item_info['total_cost'],
                        'match_score': item_info['match_score']
                    }
            
            extracted_file = os.path.join(output_dir, f"extracted_items_{timestamp}.json")
            with open(extracted_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_items, f, indent=2, ensure_ascii=False)
            
            # Save shopping list
            shopping_list = self.create_shopping_list(grocery_prices)
            shopping_file = os.path.join(output_dir, f"shopping_list_{timestamp}.json")
            with open(shopping_file, 'w', encoding='utf-8') as f:
                json.dump(shopping_list, f, indent=2, ensure_ascii=False)
            
            # Save summary
            summary = {
                'timestamp': grocery_prices['timestamp'],
                'total_items': grocery_prices['total_items'],
                'items_with_prices': grocery_prices['summary']['items_with_prices'],
                'total_estimated_cost': grocery_prices['summary']['total_estimated_cost'],
                'platforms_used': grocery_prices['summary']['platforms_used'],
                'best_overall_platform': shopping_list.get('best_platform', 'N/A'),
                'best_overall_cost': shopping_list.get('total_cost', 0)
            }
            
            summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # Save CSV version of extracted items
            if extracted_items:
                csv_file = os.path.join(output_dir, f"extracted_items_{timestamp}.csv")
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['item_name', 'quantity', 'unit', 'best_platform', 'best_product', 
                                'best_brand', 'best_variant', 'unit_price', 'total_cost', 'match_score']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for item_name, item_info in extracted_items.items():
                        row = {'item_name': item_name}
                        row.update(item_info)
                        writer.writerow(row)
            
            # Results saved successfully
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def create_shopping_list(self, grocery_prices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an optimized shopping list with platform recommendations
        
        Args:
            grocery_prices: Dictionary with prices for all items
            
        Returns:
            Optimized shopping list
        """
        # Creating optimized shopping list
        
        shopping_list = {
            'timestamp': datetime.now().isoformat(),
            'total_estimated_cost': grocery_prices['summary']['total_estimated_cost'],
            'platforms': {},
            'items_by_platform': {},
            'recommendations': []
        }
        
        # Group items by platform
        for item, info in grocery_prices['items'].items():
            if info['best_platform']:
                platform = info['best_platform']
                
                if platform not in shopping_list['platforms']:
                    shopping_list['platforms'][platform] = {
                        'total_cost': 0,
                        'item_count': 0,
                        'items': []
                    }
                
                shopping_list['platforms'][platform]['total_cost'] += info['total_cost']
                shopping_list['platforms'][platform]['item_count'] += 1
                shopping_list['platforms'][platform]['items'].append({
                    'item': item,
                    'quantity': info['quantity'],
                    'unit_price': info['unit_price'],
                    'total_cost': info['total_cost'],
                    'product': info['best_product'],
                    'brand': info['best_brand']
                })
        
        # Create recommendations
        if shopping_list['platforms']:
            # Find cheapest platform overall
            cheapest_platform = min(shopping_list['platforms'].items(), 
                                  key=lambda x: x[1]['total_cost'])
            
            shopping_list['recommendations'].append({
                'type': 'cheapest_platform',
                'platform': cheapest_platform[0],
                'total_cost': cheapest_platform[1]['total_cost'],
                'savings': 'Compare with other platforms'
            })
        
        return shopping_list
    
    def save_results(self, grocery_prices: Dict[str, Any], shopping_list: Dict[str, Any], 
                    output_dir: str = "results"):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save grocery prices
        prices_file = os.path.join(output_dir, f"grocery_prices_{timestamp}.json")
        with open(prices_file, 'w', encoding='utf-8') as f:
            json.dump(grocery_prices, f, indent=2, ensure_ascii=False)
        
        # Save shopping list
        shopping_file = os.path.join(output_dir, f"shopping_list_{timestamp}.json")
        with open(shopping_file, 'w', encoding='utf-8') as f:
            json.dump(shopping_list, f, indent=2, ensure_ascii=False)
        
        # Results saved successfully
        
        return prices_file, shopping_file

def main():
    """Test the price scraper"""
    scraper = PriceScraper()
    
    # Test with sample grocery list
    test_grocery_list = {
        'milk': 2,
        'bread': 1,
        'butter': 1,
        'oil': 1
    }
    
    grocery_prices = scraper.scrape_grocery_list_prices(test_grocery_list)
    shopping_list = scraper.create_shopping_list(grocery_prices)
    
    # Save results
    scraper.save_results(grocery_prices, shopping_list)

if __name__ == "__main__":
    main()
