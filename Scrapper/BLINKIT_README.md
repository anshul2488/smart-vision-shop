# Blinkit Scraper

This module provides a scraper for Blinkit (formerly Grofers) that extracts product information from their API.

## Features

- **API-based scraping**: Uses Blinkit's JSON API instead of HTML parsing
- **Comprehensive product data**: Extracts name, price, brand, variant, rating, reviews, inventory, and more
- **Multiple output formats**: Saves data in both JSON and CSV formats
- **Error handling**: Robust error handling and logging
- **Rate limiting**: Includes delays to be respectful to the API

## Usage

### Command Line Interface

```bash
# Scrape Blinkit for a specific item
python main.py "curd" --platform blinkit

# Scrape with custom output directory
python main.py "milk" --platform blinkit --output-dir blinkit_results

# Get help
python main.py --help
```

### Programmatic Usage

```python
from scraper.blinkit_scraper import BlinkitScraper

# Initialize scraper
scraper = BlinkitScraper()

# Scrape products
products = scraper.scrape_blinkit("curd", "output")

# Process results
for product in products:
    print(f"Product: {product['name']}")
    print(f"Price: {product['price']}")
    print(f"Brand: {product['brand']}")
    print(f"Variant: {product['variant']}")
    print(f"Rating: {product['rating']}")
    print(f"ETA: {product['eta']}")
    print("---")
```

## API Response Structure

The Blinkit scraper expects API responses in the following format:

```json
{
  "is_success": true,
  "response": {
    "snippets": [
      {
        "data": {
          "identity": {"id": "613806"},
          "name": {"text": "Chitale Pouch Curd"},
          "variant": {"text": "400 g"},
          "normal_price": {"text": "₹37"},
          "brand_name": {"text": "Chitale"},
          "rating": {
            "type": "bar",
            "bar": {
              "value": 4.3,
              "title": {"text": "(3,219)"}
            }
          },
          "image": {"url": "https://cdn.grofers.com/..."},
          "inventory": 23,
          "merchant_id": "38724",
          "merchant_type": "express",
          "eta_tag": {"title": {"text": "earliest"}}
        }
      }
    ]
  }
}
```

## Extracted Data Fields

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Product name | "Chitale Pouch Curd" |
| `price` | Product price | "₹37" |
| `variant` | Product size/variant | "400 g" |
| `brand` | Brand name | "Chitale" |
| `rating` | Product rating | "4.3" |
| `review_count` | Number of reviews | "3,219" |
| `product_url` | Blinkit deeplink URL | "grofers://pdp?product_id=613806" |
| `image_url` | Product image URL | "https://cdn.grofers.com/..." |
| `product_id` | Unique product ID | "613806" |
| `inventory` | Available stock | 23 |
| `merchant_id` | Merchant identifier | "38724" |
| `merchant_type` | Merchant type | "express" |
| `eta` | Estimated delivery time | "earliest" |
| `scraped_at` | Timestamp of scraping | "2024-01-08T14:13:39" |

## Output Files

The scraper generates two output files:

- **`blinkit_data.json`**: Complete product data in JSON format
- **`blinkit_data.csv`**: Product data in CSV format for spreadsheet applications

## Configuration

### API Endpoint

The scraper uses the following API endpoint:
```
https://blinkit.com/api/v1/search?q={item_name}&limit=50&page=1
```

### Headers

The scraper includes appropriate headers to mimic a browser request:
- User-Agent
- Accept headers
- Referer and Origin
- Security headers

## Error Handling

The scraper includes comprehensive error handling for:
- Network connectivity issues
- API rate limiting
- Invalid responses
- Missing data fields
- File I/O errors

## Rate Limiting

To be respectful to Blinkit's servers, the scraper includes:
- Random delays between requests (1-3 seconds)
- Proper session management
- Timeout handling

## Testing

Run the test script to verify the scraper functionality:

```bash
python test_blinkit.py
```

## Limitations

1. **API Access**: The scraper relies on Blinkit's public API, which may have restrictions
2. **Rate Limiting**: Too many requests may result in temporary blocks
3. **Data Availability**: Some products may not have complete information
4. **API Changes**: Blinkit may change their API structure, requiring updates

## Dependencies

- `requests`: For HTTP requests
- `json`: For JSON parsing (built-in)
- `csv`: For CSV output (built-in)
- `logging`: For logging (built-in)

## Notes

- The scraper is designed to work with Blinkit's current API structure
- Results may vary based on location and availability
- Some products may not be available in all regions
- The scraper respects Blinkit's terms of service by using reasonable delays
