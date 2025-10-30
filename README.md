# 🛒 SmartVisionShop - Advanced Grocery Price Comparison System

## 🎯 **What This System Does**

This is a **complete grocery price comparison system** that takes handwritten grocery lists and provides real-time price comparisons across **4 major platforms** with a modern web interface!

### **Complete Workflow:**
1. **📸 OCR Processing**: Handwritten list → Text (Advanced AI preprocessing)
2. **🔍 Item Parsing**: Text → Items and quantities  
3. **💰 Price Scraping**: Items → Real-time prices from Amazon, Zepto, BigBasket, and JioMart
4. **🎯 Price Comparison**: Interactive web interface with best deals and recommendations
5. **🛒 Shopping Integration**: Direct links to purchase from each platform

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.10 or 3.11
- pip (Python package manager)
- Git (for cloning the repository)

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd project_pipeline
```

### **2. Set Up Virtual Environment**

#### **Windows:**
```bash
# Create virtual environment
python -m venv btech311

# Activate virtual environment
btech311\Scripts\activate

# Verify activation (should show (btech311) in prompt)
```

#### **Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv btech311

# Activate virtual environment
source btech311/bin/activate

# Verify activation (should show (btech311) in prompt)
```

### **3. Install Dependencies**

#### **Core Dependencies:**
```bash
# Install main requirements
pip install -r requirements.txt

# Install Flask web app dependencies
pip install -r requirements_flask.txt
```

#### **OCR Dependencies (Optional - for advanced handwritten OCR):**
```bash
# Install OCR-specific requirements
pip install -r models/requirements_ocr.txt

# Install additional OCR libraries
pip install easyocr pytesseract
```

#### **System Dependencies:**

**Windows:**
- Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Add Tesseract to your PATH

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

**macOS:**
```bash
brew install tesseract
```

### **4. Start the Application**

#### **Development Mode:**
```bash
# Start the Flask development server
python app.py

# Or use the dedicated server script
python run_server.py
```

#### **Production Mode:**
```bash
# Using Gunicorn (recommended for production)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or using Waitress (Windows-friendly)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### **5. Access the Application**
```
# Open your browser and go to:
http://localhost:5000

# For production (if running on server):
http://your-server-ip:5000
```

## 🏗️ **Production Deployment**

### **Docker Deployment (Recommended)**

#### **1. Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements_flask.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements_flask.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

#### **2. Build and Run Docker Container:**
```bash
# Build the Docker image
docker build -t smartvisionshop .

# Run the container
docker run -d -p 5000:5000 --name smartvisionshop-app smartvisionshop

# Check if running
docker ps
```

### **Cloud Deployment (AWS/GCP/Azure)**

#### **AWS EC2 Deployment:**
```bash
# 1. Launch EC2 instance (Ubuntu 20.04 LTS)
# 2. Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install dependencies
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv nginx git

# 4. Clone and setup
git clone <repository-url>
cd project_pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements_flask.txt gunicorn

# 5. Configure Nginx
sudo nano /etc/nginx/sites-available/smartvisionshop
```

**Nginx Configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# 6. Enable site and start services
sudo ln -s /etc/nginx/sites-available/smartvisionshop /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# 7. Start application with systemd
sudo nano /etc/systemd/system/smartvisionshop.service
```

**Systemd Service:**
```ini
[Unit]
Description=SmartVisionShop Web Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/project_pipeline
Environment=PATH=/home/ubuntu/project_pipeline/venv/bin
ExecStart=/home/ubuntu/project_pipeline/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 8. Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable smartvisionshop
sudo systemctl start smartvisionshop
sudo systemctl status smartvisionshop
```

### **Environment Variables**

Create a `.env` file for production configuration:
```bash
# Production settings
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here

# Database settings (if using database)
DATABASE_URL=sqlite:///production.db

# API keys (if needed)
AMAZON_API_KEY=your-amazon-api-key
BIGBASKET_API_KEY=your-bigbasket-api-key

# Logging
LOG_LEVEL=ERROR
```

## 🔧 **Configuration**

### **Application Settings**
```python
# In app.py, you can modify:
DEBUG = False  # Set to True for development
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5000  # Port number
```

### **Scraping Settings**
```python
# In price_scraper.py:
MAX_RESULTS_PER_ITEM = 5  # Number of products per platform
REQUEST_TIMEOUT = 30  # Timeout in seconds
RETRY_ATTEMPTS = 3  # Number of retry attempts
```

### **OCR Settings**
```python
# In ocr_processor.py:
CONFIDENCE_THRESHOLD = 0.5  # OCR confidence threshold
MAX_IMAGE_SIZE = (1920, 1080)  # Maximum image dimensions
```

## 📁 **Project Structure**

```
project_pipeline/
├── app.py                           # Flask web application
├── price_scraper.py                 # Multi-platform price scraping
├── ocr_processor.py                 # Advanced OCR for handwritten text
├── grocery_pipeline.py              # Main pipeline coordinator
├── run_handwritten_ocr.py           # Handwritten OCR launcher script
├── run_server.py                    # Production server launcher
├── Frontend/                        # Web interface
│   ├── index.html                   # Main web page
│   ├── single-search.html           # Single item search page
│   ├── bulk-search.html             # Bulk search page
│   ├── app.js                       # Main frontend JavaScript
│   ├── single-search.js             # Single search JavaScript
│   ├── bulk-search.js               # Bulk search JavaScript
│   └── styles.css                   # Modern UI styling
├── models/                          # Handwritten OCR model system
│   ├── handwritten_ocr_model.py     # CRNN model architecture
│   ├── train_handwritten_ocr.py     # Training script
│   ├── inference_handwritten_ocr.py # Inference script
│   ├── evaluate_handwritten_ocr.py  # Evaluation script
│   ├── requirements_ocr.txt         # OCR dependencies
│   └── HANDWRITTEN_OCR_README.md    # OCR documentation
├── Scrapper/                        # Platform scrapers
│   ├── amazon_scraper.py            # Amazon India scraper
│   ├── zepto_scraper_advanced.py    # Zepto scraper
│   ├── bigbasket_scraper.py         # BigBasket scraper
│   ├── jiomart_scraper.py           # JioMart scraper
│   └── utils.py                     # Scraper utilities
├── dataset/                         # Training data
│   └── train_data/                  # Handwritten grocery list images + text
├── pipeline_results/                # Output results
├── temp_output/                     # Temporary scraper outputs
├── requirements.txt                 # Core dependencies
├── requirements_flask.txt           # Web app dependencies
├── README.md                        # This file
└── .gitignore                      # Git ignore rules
```

## 🎯 **Usage Examples**

### **Web Application (Recommended)**
```bash
# Start the web server
python app.py

# Open browser to http://localhost:5000
# Upload image or type grocery list
# Get instant price comparison
```

### **Command Line Interface**
```bash
# Process handwritten list
python grocery_pipeline.py my_grocery_list.jpg

# Test with sample data
python test_pipeline.py

# Test individual scrapers
python -c "from Scrapper.bigbasket_scraper import BigBasketScraper; scraper = BigBasketScraper(); print(scraper.search_products('milk', 3))"
```

### **API Usage**
```python
# Test OCR only
from ocr_processor import OCRProcessor
processor = OCRProcessor()
grocery_dict = processor.process_grocery_list("image.jpg")

# Test price scraping only
from price_scraper import PriceScraper
scraper = PriceScraper()
prices = scraper.scrape_grocery_list_prices(grocery_dict)

# Test individual platform scrapers
from Scrapper.zepto_scraper_advanced import AdvancedZeptoScraper
zepto = AdvancedZeptoScraper()
results = zepto.search_products('bread', 5)
```

## 📊 **Web Interface Features**

### **Modern UI Components**
- **📱 Responsive Design**: Works on desktop, tablet, and mobile
- **🎨 Beautiful Interface**: Modern card-based layout with smooth animations
- **⚡ Real-time Updates**: Live price comparison with loading animations
- **🎯 Interactive Elements**: Hover effects, click-to-remove items, smooth transitions

### **Price Comparison Display**
- **4 Platform Cards**: Amazon, Zepto, BigBasket, JioMart side-by-side
- **Best Deal Highlighting**: Cheapest platform automatically highlighted
- **Price Breakdown**: Shows unit prices, quantities, and total calculations
- **Product Details**: Brand, rating, variant, delivery time, match scores
- **Direct Shopping**: One-click buttons to visit each platform

### **Input Methods**
- **📸 Image Upload**: Take photo of handwritten grocery list
- **⌨️ Manual Input**: Type or paste grocery items
- **📋 Sample Data**: Load sample grocery list for testing
- **🔄 Reset Function**: Clear all inputs and start fresh

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

## 🔧 **Advanced Configuration**

### **OCR Settings**
- **EasyOCR**: Primary OCR engine (better for handwritten text)
- **Tesseract**: Fallback OCR engine
- **Confidence threshold**: 0.5 (adjustable)

### **Price Scraping**
- **Platforms**: Amazon India, Zepto, BigBasket, JioMart
- **Max results per item**: 5 (adjustable)
- **Timeout**: 30 seconds per request
- **Anti-detection**: User agent rotation, realistic headers, fallback systems

### **Output Settings**
- **Output directory**: `pipeline_results/`
- **File formats**: JSON
- **Timestamp**: Automatic

## 🐛 **Troubleshooting**

### **Common Issues**

#### **OCR Issues**
```bash
# Install EasyOCR
pip install easyocr

# Install Tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

#### **Scraping Issues**
- Make sure internet connection is stable
- Check if platforms are accessible
- Try with sample data first
- Check firewall settings

#### **Import Errors**
```bash
# Make sure you're in the project_pipeline directory
cd project_pipeline

# Install all requirements
pip install -r requirements.txt -r requirements_flask.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### **Port Already in Use**
```bash
# Find process using port 5000
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows

# Kill the process or use different port
python app.py --port 5001
```

### **Performance Optimization**

#### **For Production:**
```bash
# Use Gunicorn with multiple workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Use Nginx as reverse proxy
# Configure Nginx to serve static files
# Enable gzip compression
```

#### **For Development:**
```bash
# Enable debug mode
export FLASK_DEBUG=1
python app.py

# Use development server with auto-reload
python run_server.py
```

## 📈 **Monitoring and Logging**

### **Application Logs**
```python
# Logs are configured in app.py
# Production: Only ERROR level logs
# Development: INFO level logs
```

### **Health Checks**
```bash
# Check if application is running
curl http://localhost:5000/health

# Check individual scrapers
python -c "from Scrapper.amazon_scraper import AmazonScraper; print(AmazonScraper().search_products('milk', 1))"
```

## 🚀 **Ready to Use!**

Your advanced grocery price comparison system is complete and ready! It provides:

1. ✅ **Advanced OCR**: Reads handwritten grocery lists with AI preprocessing
2. ✅ **Smart Parsing**: Extracts items and quantities with confidence scoring
3. ✅ **4-Platform Scraping**: Real-time prices from Amazon, Zepto, BigBasket, JioMart
4. ✅ **Modern Web Interface**: Beautiful, responsive UI with real-time updates
5. ✅ **Price Comparison**: Side-by-side comparison with best deal highlighting
6. ✅ **Direct Shopping**: One-click links to purchase from each platform
7. ✅ **Fallback Systems**: Reliable operation even when scraping is blocked
8. ✅ **Production Ready**: Docker support, cloud deployment guides, monitoring

**Start using it now:**
```bash
# Development
python app.py
# Open http://localhost:5000 in your browser

# Production
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Command Line
python grocery_pipeline.py your_grocery_list.jpg
```

**Perfect for:**
- 🛒 **Smart Shopping**: Find the best deals across all major platforms
- 💰 **Budget Planning**: Compare total costs and save money
- ⚡ **Quick Decisions**: Get instant price comparisons
- 📱 **Mobile Shopping**: Use on any device with responsive design
- 🏢 **Business Use**: Deploy in production with monitoring and scaling

**Ready for production use with real grocery shopping!** 🎉

## 📞 **Support**

For issues, questions, or contributions:
- Check the troubleshooting section above
- Review the project structure and configuration
- Test with sample data first
- Check logs for error messages

**Happy Shopping!** 🛒✨
