# Deployment Guide for Render

## Prerequisites
- Render account
- Git repository with this code

## Deployment Steps

### 1. Connect to Render
1. Go to https://render.com
2. Connect your Git repository
3. Create a new Web Service

### 2. Configure Build Settings
- **Build Command**: `pip install -r requirements_flask.txt`
- **Start Command**: `gunicorn -w 4 -b 0.0.0.0:$PORT app:app`
- **Environment**: Python 3
- **Python Version**: 3.11.0

### 3. Environment Variables (Optional)
No environment variables required by default.

### 4. Important Notes
- The app uses lazy loading for models to reduce startup time
- First request may be slower as models load
- Ensure `requirements_flask.txt` has all dependencies
- Frontend files are served from `Frontend/` directory

### 5. File Structure Required for Deployment
```
project_pipeline/
├── app.py                    # Main Flask application
├── grocery_ocr_llm_model.py  # OCR and LLM processing
├── price_scraper.py          # Price scraping logic
├── specialized_preprocessing.py  # Image preprocessing
├── handwritten_ocr_integration.py  # Handwritten OCR
├── requirements_flask.txt    # Python dependencies
├── Frontend/                 # Frontend files
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── Scrapper/                 # Price scrapers
│   ├── amazon_scraper.py
│   ├── utils.py
│   └── ...
└── models/                   # Trained models (if used)
    ├── best_model.pth
    └── handwritten_ocr_model.py
```

### 6. Troubleshooting
- If build fails, check `requirements_flask.txt` for all dependencies
- Ensure all Python files are in the root or properly referenced
- Check Render logs for specific error messages

