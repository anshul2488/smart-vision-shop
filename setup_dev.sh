#!/bin/bash
# SmartVisionShop Development Setup Script
# This script sets up the application for development

set -e  # Exit on any error

echo "🚀 SmartVisionShop Development Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is installed
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 1 ]]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Check if pip is installed
print_status "Checking pip..."
if command -v pip3 &> /dev/null; then
    print_success "pip3 found"
else
    print_error "pip3 is not installed"
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements_flask.txt

# Install development dependencies
print_status "Installing development dependencies..."
pip install pytest pytest-cov black flake8 mypy

# Install OCR dependencies
print_status "Installing OCR dependencies..."
pip install easyocr pytesseract

print_success "All dependencies installed"

# Install system dependencies for OCR
print_status "Installing system dependencies for OCR..."
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr libtesseract-dev
    print_success "OCR dependencies installed (Ubuntu/Debian)"
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    sudo yum install -y tesseract tesseract-devel
    print_success "OCR dependencies installed (CentOS/RHEL)"
elif command -v brew &> /dev/null; then
    # macOS
    brew install tesseract
    print_success "OCR dependencies installed (macOS)"
else
    print_warning "Could not install OCR dependencies automatically. Please install Tesseract manually."
fi

# Create development configuration
print_status "Creating development configuration..."
cat > .env << EOF
# Development settings
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production

# Logging
LOG_LEVEL=INFO

# Server settings
HOST=127.0.0.1
PORT=5000
EOF
print_success "Development configuration created"

# Create development startup script
print_status "Creating development startup script..."
cat > start_dev.sh << 'EOF'
#!/bin/bash
# SmartVisionShop Development Startup Script

echo "🚀 Starting SmartVisionShop in Development Mode..."

# Activate virtual environment
source venv/bin/activate

# Set development environment variables
export FLASK_ENV=development
export FLASK_DEBUG=True

# Start the application with Flask development server
python app.py
EOF

chmod +x start_dev.sh
print_success "Development startup script created"

# Create test script
print_status "Creating test script..."
cat > test_app.sh << 'EOF'
#!/bin/bash
# SmartVisionShop Test Script

echo "🧪 Testing SmartVisionShop..."

# Activate virtual environment
source venv/bin/activate

# Run tests
echo "Running unit tests..."
python -m pytest tests/ -v --cov=. --cov-report=html

# Test individual components
echo "Testing OCR processor..."
python -c "
from ocr_processor import OCRProcessor
processor = OCRProcessor()
print('✅ OCR Processor initialized successfully')
"

echo "Testing price scraper..."
python -c "
from price_scraper import PriceScraper
scraper = PriceScraper()
print('✅ Price Scraper initialized successfully')
"

echo "Testing individual scrapers..."
python -c "
from Scrapper.amazon_scraper import AmazonScraper
from Scrapper.bigbasket_scraper import BigBasketScraper
from Scrapper.zepto_scraper_advanced import AdvancedZeptoScraper
from Scrapper.jiomart_scraper import JioMartScraper

scrapers = [
    ('Amazon', AmazonScraper()),
    ('BigBasket', BigBasketScraper()),
    ('Zepto', AdvancedZeptoScraper()),
    ('JioMart', JioMartScraper())
]

for name, scraper in scrapers:
    try:
        results = scraper.search_products('milk', 1)
        print(f'✅ {name}: Working ({len(results)} results)')
    except Exception as e:
        print(f'❌ {name}: Error - {str(e)[:50]}...')
"

echo "✅ All tests completed"
EOF

chmod +x test_app.sh
print_success "Test script created"

# Create code quality script
print_status "Creating code quality script..."
cat > check_code.sh << 'EOF'
#!/bin/bash
# SmartVisionShop Code Quality Check Script

echo "🔍 Checking code quality..."

# Activate virtual environment
source venv/bin/activate

# Run Black formatter
echo "Running Black formatter..."
black --check .

# Run Flake8 linter
echo "Running Flake8 linter..."
flake8 .

# Run MyPy type checker
echo "Running MyPy type checker..."
mypy . --ignore-missing-imports

echo "✅ Code quality check completed"
EOF

chmod +x check_code.sh
print_success "Code quality script created"

# Create sample data script
print_status "Creating sample data script..."
cat > create_sample_data.sh << 'EOF'
#!/bin/bash
# SmartVisionShop Sample Data Creation Script

echo "📝 Creating sample data..."

# Create sample grocery list
cat > sample_grocery_list.json << 'SAMPLE'
{
  "milk": "2",
  "bread": "1",
  "butter": "1",
  "rice": "5",
  "oil": "1",
  "eggs": "12",
  "tomato": "1",
  "onion": "1"
}
SAMPLE

# Create sample image directory
mkdir -p sample_images

echo "✅ Sample data created"
echo "   - sample_grocery_list.json: Sample grocery list"
echo "   - sample_images/: Directory for sample images"
EOF

chmod +x create_sample_data.sh
print_success "Sample data script created"

# Final instructions
print_success "Development setup completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Start the development server:"
echo "   ./start_dev.sh"
echo ""
echo "2. Run tests:"
echo "   ./test_app.sh"
echo ""
echo "3. Check code quality:"
echo "   ./check_code.sh"
echo ""
echo "4. Create sample data:"
echo "   ./create_sample_data.sh"
echo ""
echo "5. Access the application:"
echo "   http://localhost:5000"
echo ""
echo "📚 Development Commands:"
echo "   - Start dev server: ./start_dev.sh"
echo "   - Run tests: ./test_app.sh"
echo "   - Check code: ./check_code.sh"
echo "   - Create samples: ./create_sample_data.sh"
echo "   - Format code: black ."
echo "   - Lint code: flake8 ."
echo "   - Type check: mypy ."
echo ""
echo "🔧 Development Tips:"
echo "   - Use 'export FLASK_DEBUG=True' for auto-reload"
echo "   - Check logs in the terminal for debugging"
echo "   - Use sample data for testing without real scraping"
echo "   - Test individual scrapers before full pipeline"
echo ""
print_success "SmartVisionShop is ready for development! 🎉"
