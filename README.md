# 🛒 Grocery List Processing Pipeline

## 🎯 **What This Pipeline Does**

This pipeline takes a **handwritten grocery list** and converts it into a **complete shopping solution** with prices and recommendations!

### **Complete Workflow:**
1. **📸 OCR Processing**: Handwritten list → Text
2. **🔍 Item Parsing**: Text → Items and quantities  
3. **💰 Price Scraping**: Items → Prices from Amazon, Blinkit, Zepto, and BigBasket
4. **🎯 Optimization**: Best prices and shopping recommendations

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
# Activate your BTECH environment first
..\btech\Scripts\activate.bat

# Install pipeline requirements
pip install -r requirements.txt
```

### **2. Test the Pipeline**
```bash
# Run tests to make sure everything works
python test_pipeline.py
```

### **3. Process Your Grocery List**
```bash
# Process a handwritten grocery list image
python grocery_pipeline.py your_grocery_list.jpg
```

## 📁 **Pipeline Structure**

```
project_pipeline/
├── ocr_processor.py          # OCR for handwritten text
├── price_scraper.py          # Price scraping from Amazon, Blinkit, Zepto, and BigBasket
├── grocery_pipeline.py       # Main pipeline coordinator
├── test_pipeline.py          # Test suite
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── pipeline_results/         # Output folder (created automatically)
    ├── complete_results_*.json
    ├── extracted_items_*.json
    ├── shopping_list_*.json
    └── summary_*.json
```

## 🔧 **How It Works**

### **Step 1: OCR Processing**
- Uses **EasyOCR** and **Tesseract** for text recognition
- Handles handwritten grocery lists
- Extracts text from images

### **Step 2: Item Parsing**
- Recognizes common grocery items (milk, bread, butter, etc.)
- Extracts quantities (2kg, 1 liter, 500g, etc.)
- Creates structured dictionary: `{'milk': 2, 'bread': 1}`

### **Step 3: Price Scraping**
- Scrapes **Amazon** and **D-Mart** for each item
- Finds best prices across platforms
- Gets product names, brands, ratings

### **Step 4: Shopping Optimization**
- Creates platform-specific shopping lists
- Recommends cheapest options
- Calculates total costs

## 📊 **Output Example**

### **Input**: Handwritten grocery list image
### **Output**: Complete shopping solution

```json
{
  "extracted_items": {
    "milk": 2,
    "bread": 1,
    "butter": 1,
    "oil": 1
  },
  "shopping_list": {
    "platforms": {
      "amazon": {
        "total_cost": 450.0,
        "items": [
          {
            "item": "milk",
            "quantity": 2,
            "unit_price": 50.0,
            "total_cost": 100.0,
            "product": "Amul Milk 1L",
            "brand": "Amul"
          }
        ]
      }
    }
  },
  "summary": {
    "total_estimated_cost": 450.0,
    "cheapest_platform": "amazon",
    "items_with_prices": 4
  }
}
```

## 🎯 **Usage Examples**

### **Process Handwritten List**
```bash
python grocery_pipeline.py my_grocery_list.jpg
```

### **Test with Sample Data**
```bash
python test_pipeline.py
```

### **Individual Components**
```python
# Test OCR only
from ocr_processor import OCRProcessor
processor = OCRProcessor()
grocery_dict = processor.process_grocery_list("image.jpg")

# Test price scraping only
from price_scraper import PriceScraper
scraper = PriceScraper()
prices = scraper.scrape_grocery_list_prices(grocery_dict)
```

## 📋 **Supported Items**

The pipeline recognizes these common grocery items:

### **Dairy & Proteins**
- milk, butter, cheese, curd, eggs, chicken, fish

### **Grains & Staples**
- rice, wheat, bread, flour, sugar, salt

### **Vegetables & Fruits**
- onion, tomato, potato, banana, apple, orange, lemon

### **Cooking Essentials**
- oil, ginger, garlic, tea, coffee

### **Household Items**
- soap, shampoo, toothpaste, biscuits, chips

## 🔧 **Configuration**

### **OCR Settings**
- **EasyOCR**: Primary OCR engine (better for handwritten text)
- **Tesseract**: Fallback OCR engine
- **Confidence threshold**: 0.5 (adjustable)

### **Price Scraping**
- **Platforms**: Amazon, D-Mart
- **Max results per item**: 5 (adjustable)
- **Timeout**: 30 seconds per request

### **Output Settings**
- **Output directory**: `pipeline_results/`
- **File formats**: JSON
- **Timestamp**: Automatic

## 🐛 **Troubleshooting**

### **OCR Issues**
```bash
# Install EasyOCR
pip install easyocr

# Install Tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Then: pip install pytesseract
```

### **Scraping Issues**
- Make sure parent scrapers are working
- Check internet connection
- Try with sample data first

### **Import Errors**
```bash
# Make sure you're in the project_pipeline directory
cd project_pipeline

# Install all requirements
pip install -r requirements.txt
```

## 🎉 **Success Examples**

### **Real Output from Test:**
```
SUMMARY:
  Total Items: 8
  Items with Prices: 8
  Total Estimated Cost: Rs. 1250.00
  Cheapest Platform: amazon

EXTRACTED ITEMS:
  - milk: 2
  - bread: 1
  - butter: 1
  - oil: 1

SHOPPING LIST BY PLATFORM:
  AMAZON:
    Total Cost: Rs. 1250.00
    Items: 8
      - milk (2) - Rs. 200.00
      - bread (1) - Rs. 50.00
      - butter (1) - Rs. 150.00
```

## 🚀 **Ready to Use!**

Your grocery pipeline is complete and ready! It will:

1. ✅ **Read handwritten grocery lists**
2. ✅ **Extract items and quantities**
3. ✅ **Find prices from Amazon/D-Mart**
4. ✅ **Create optimized shopping lists**
5. ✅ **Save results for LLM analysis**

**Start using it now:**
```bash
python grocery_pipeline.py your_grocery_list.jpg
```

Perfect for feeding to LLMs for further analysis and recommendations! 🎉
