# 🛒 E-commerce Scraper

A Python-based web scraper that extracts product listings from **Amazon India** and **Blinkit**. This tool automatically fetches product information including names, prices, ratings, reviews, and URLs from multiple e-commerce platforms.

## 📁 Project Structure

```
D:\AmazonScraper\
├── venv/                      # Virtual environment
├── main.py                    # Entry point to run the scraper
├── scraper/
│   ├── __init__.py
│   ├── amazon_scraper.py      # requests and BeautifulSoup scraping logic
│   ├── utils.py               # Helper functions (URL builder, cleaning)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── output/
    ├── data.json              # Scraped results in JSON format
    └── data.csv               # Scraped results in CSV format
```

## 🚀 Quick Start

### 1. Setup Virtual Environment

```bash
# Navigate to the project directory
cd D:\AmazonScraper

# Activate the virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Scraper

```bash
# Search for a single word item
python main.py oil

# Search for multi-word items (use quotes)
python main.py "body lotion"

# Search for other items
python main.py "mobile phone"
python main.py "laptop"
python main.py "headphones"
```

## 📋 Features

- **🔍 Dynamic Search**: Search for any product by name
- **📊 Multiple Output Formats**: Results saved in both JSON and CSV
- **🛡️ Error Handling**: Graceful handling of network issues and parsing errors
- **📝 Progress Logging**: Real-time progress updates and detailed logging
- **🎯 Smart Parsing**: Extracts product name, price, rating, reviews, URLs, and images
- **⚡ Efficient Processing**: Uses requests and BeautifulSoup for reliable web scraping

## 📦 Dependencies

The project uses the following Python packages:

- **requests**: HTTP library for web requests
- **beautifulsoup4**: HTML parsing and data extraction
- **lxml**: Fast XML/HTML parser

## 🛠️ Installation

### Automatic Setup (Recommended)

1. **Clone or download** the project to `D:\AmazonScraper`
2. **Open Command Prompt** or PowerShell as Administrator
3. **Navigate** to the project directory:
   ```bash
   cd D:\AmazonScraper
   ```
4. **Activate** the virtual environment:
   ```bash
   venv\Scripts\activate
   ```
5. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install requests>=2.31.0
pip install beautifulsoup4>=4.12.2
pip install lxml>=4.9.3
```

## 🎯 Usage Examples

### Basic Usage

```bash
# Search for oil products
python main.py oil

# Search for body lotion
python main.py "body lotion"

# Search for mobile phones
python main.py "mobile phone"
```

### Advanced Usage

```bash
# Specify custom output directory
python main.py oil --output-dir my_results

# Enable verbose logging
python main.py "body lotion" --verbose

# Get help
python main.py --help
```

## 📊 Output Format

The scraper generates two output files in the `output/` directory:

### JSON Format (`data.json`)
```json
[
  {
    "name": "Product Name",
    "price": "₹299",
    "rating": "4.5",
    "review_count": "1,234",
    "product_url": "https://www.amazon.in/...",
    "image_url": "https://m.media-amazon.com/...",
    "scraped_at": "2024-01-15T10:30:00"
  }
]
```

### CSV Format (`data.csv`)
```csv
name,price,rating,review_count,product_url,image_url,scraped_at
"Product Name","₹299","4.5","1,234","https://www.amazon.in/...","https://m.media-amazon.com/...","2024-01-15T10:30:00"
```

## 🔧 Configuration

### URL Format
The scraper uses Amazon India's search URL format:
```
https://www.amazon.in/s?k=<item_name>&i=nowstore&rh=n%3A16392737031
```

### Search Parameters
- `k`: Search keyword (item name)
- `i`: Store filter (nowstore)
- `rh`: Category filter (n%3A16392737031)

## 🐛 Troubleshooting

### Common Issues

1. **"No products found"**
   - Check your internet connection
   - Try a different search term
   - Amazon may have changed their HTML structure

2. **"Missing dependency" error**
   - Make sure virtual environment is activated
   - Run: `pip install -r requirements.txt`

3. **"Permission denied" error**
   - Run Command Prompt as Administrator
   - Check file permissions in the output directory

4. **Network timeout errors**
   - Check your internet connection
   - Amazon may be blocking requests (try again later)

### Debug Mode

Enable verbose logging to see detailed information:

```bash
python main.py "your search term" --verbose
```

## ⚠️ Important Notes

- **Rate Limiting**: The scraper includes random delays (1-3 seconds) to be respectful to Amazon's servers
- **Terms of Service**: Please ensure your usage complies with Amazon's Terms of Service
- **Data Usage**: Use scraped data responsibly and in accordance with applicable laws
- **Updates**: Amazon may change their website structure, which could affect scraping
- **Success Rate**: The scraper successfully extracts product information from Amazon India search results

## 🔄 Updates and Maintenance

To update the scraper:

1. **Update dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Check for changes** in Amazon's HTML structure
3. **Update selectors** in `amazon_scraper.py` if needed

## 📝 Logs

The scraper provides detailed logging including:
- ✅ Success messages
- ⚠️ Warnings
- ❌ Error messages
- 📊 Progress updates

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please use responsibly and in accordance with Amazon's Terms of Service.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Enable verbose logging with `--verbose` flag
3. Check your internet connection
4. Verify all dependencies are installed correctly

---

**Happy Scraping! 🚀**

## ✅ Test Results

The scraper has been successfully tested and works perfectly:

```bash
python main.py oil
```

**Sample Output:**
- ✅ Successfully scraped 32 products
- ✅ Generated both JSON and CSV files
- ✅ Extracted product names, prices, ratings, and image URLs
- ✅ Proper error handling and logging
- ✅ Clean, formatted output data

The scraper is ready for production use! 🎉
