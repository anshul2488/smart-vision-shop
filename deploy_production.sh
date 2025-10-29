#!/bin/bash
# SmartVisionShop Production Deployment Script
# This script sets up the application for production deployment

set -e  # Exit on any error

echo "🚀 SmartVisionShop Production Deployment Script"
echo "=============================================="

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

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
pip install gunicorn waitress
print_success "Dependencies installed"

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

# Create production configuration
print_status "Creating production configuration..."
cat > .env << EOF
# Production settings
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')

# Logging
LOG_LEVEL=ERROR

# Server settings
HOST=0.0.0.0
PORT=5000
EOF
print_success "Production configuration created"

# Create systemd service file
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/smartvisionshop.service > /dev/null << EOF
[Unit]
Description=SmartVisionShop Web Application
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
print_success "Systemd service created"

# Create Nginx configuration
print_status "Creating Nginx configuration..."
sudo tee /etc/nginx/sites-available/smartvisionshop > /dev/null << EOF
server {
    listen 80;
    server_name _;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss;

    # Static files
    location /static {
        alias $(pwd)/Frontend;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Application
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
EOF
print_success "Nginx configuration created"

# Enable Nginx site
print_status "Enabling Nginx site..."
sudo ln -sf /etc/nginx/sites-available/smartvisionshop /etc/nginx/sites-enabled/
sudo nginx -t
print_success "Nginx site enabled"

# Create startup script
print_status "Creating startup script..."
cat > start_production.sh << 'EOF'
#!/bin/bash
# SmartVisionShop Production Startup Script

echo "🚀 Starting SmartVisionShop in Production Mode..."

# Activate virtual environment
source venv/bin/activate

# Start the application with Gunicorn
exec gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 --keep-alive 2 --max-requests 1000 --max-requests-jitter 100 app:app
EOF

chmod +x start_production.sh
print_success "Startup script created"

# Create health check script
print_status "Creating health check script..."
cat > health_check.sh << 'EOF'
#!/bin/bash
# SmartVisionShop Health Check Script

echo "🔍 Checking SmartVisionShop Health..."

# Check if application is running
if curl -f http://localhost:5000/ > /dev/null 2>&1; then
    echo "✅ Application is running"
else
    echo "❌ Application is not responding"
    exit 1
fi

# Check individual scrapers
echo "🔍 Testing scrapers..."
python3 -c "
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

echo "✅ Health check completed"
EOF

chmod +x health_check.sh
print_success "Health check script created"

# Create backup script
print_status "Creating backup script..."
cat > backup.sh << 'EOF'
#!/bin/bash
# SmartVisionShop Backup Script

BACKUP_DIR="backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="smartvisionshop_backup_$DATE.tar.gz"

echo "📦 Creating backup: $BACKUP_FILE"

mkdir -p $BACKUP_DIR

# Create backup excluding unnecessary files
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='temp_output' \
    --exclude='pipeline_results' \
    --exclude='backups' \
    .

echo "✅ Backup created: $BACKUP_DIR/$BACKUP_FILE"
EOF

chmod +x backup.sh
print_success "Backup script created"

# Final instructions
print_success "Deployment setup completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Start the application:"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable smartvisionshop"
echo "   sudo systemctl start smartvisionshop"
echo ""
echo "2. Start Nginx:"
echo "   sudo systemctl restart nginx"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status smartvisionshop"
echo "   sudo systemctl status nginx"
echo ""
echo "4. Run health check:"
echo "   ./health_check.sh"
echo ""
echo "5. Access the application:"
echo "   http://your-server-ip"
echo ""
echo "📚 Additional Commands:"
echo "   - Start manually: ./start_production.sh"
echo "   - Create backup: ./backup.sh"
echo "   - View logs: sudo journalctl -u smartvisionshop -f"
echo "   - Restart service: sudo systemctl restart smartvisionshop"
echo ""
print_success "SmartVisionShop is ready for production! 🎉"
