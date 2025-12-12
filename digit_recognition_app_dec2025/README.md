# Digit Recognition App - December 2025

A standalone Streamlit application for recognizing handwritten digits from uploaded images using an ensemble of CNN and OCR models.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Compile Styles

```bash
streamlit-sal compile
```

### 3. Run the App

**Option A: Using startup script (Mac/Linux)**
```bash
./start-app.sh
```

**Option B: Using startup script (Windows)**
```cmd
start-app.bat
```

**Option C: Direct command**
```bash
streamlit run digit_recognition_app.py
```

## Features

- **Single-digit recognition**: Recognizes individual digits using ensemble of CNN models (SVHN, USPS) and OCR engines
- **Multi-digit recognition**: Automatically detects and recognizes sequences of multiple digits
- **Clear processing indicators**: Visual badges show whether images were processed as single-digit or multi-digit
- **Model ensemble**: Combines predictions from multiple models for improved accuracy

## Model Configuration

- **SVHN Model**: ✅ Used in ensemble
- **USPS Model**: ✅ Used in ensemble  
- **EMNIST Model**: ⚠️ Loaded but excluded from predictions (files kept for reference)
- **OCR Engines**: EasyOCR, Tesseract, PaddleOCR (optional, loaded if available)
- **YOLO Model**: Used for multi-digit detection (optional)

## File Structure

```
digit_recognition_app_dec2025/
├── digit_recognition_app.py    # Main application entry point
├── digit_recognition.py         # Core recognition engine
├── digit_models_loader.py       # Model loading functions
├── components.py                # UI components
├── constants.py                 # Configuration constants
├── utils.py                     # Utility functions
├── api_requests.py              # API request handler (legacy, not used)
├── requirements.txt             # Python dependencies
├── start-app.sh                 # Mac/Linux startup script
├── start-app.bat                # Windows startup script
├── .streamlit_sal               # Style compilation config
├── models/                      # Model files directory
│   ├── svhn_model.h5           # SVHN CNN model
│   ├── usps_model.h5           # USPS CNN model
│   ├── emnist_model.h5         # EMNIST model (excluded from predictions)
│   └── yolov8n.pt               # YOLO v8 model
├── assets/                      # Static assets
│   ├── seagull-logo.png        # Application logo
│   ├── favicon.png             # Browser favicon
│   └── empty_chat.svg          # Empty state image
└── styles/                      # SCSS stylesheets
    ├── main.scss
    ├── variables.scss
    ├── icons.scss
    └── sal-stylesheet.css       # Compiled CSS (auto-generated)
```

## Usage

1. **Upload Images**: Use the file uploader to select one or more images containing handwritten digits
2. **Automatic Processing**: The app automatically processes uploaded images:
   - Detects if image contains single or multiple digits
   - Processes accordingly using appropriate methods
   - Displays results with confidence scores
3. **View Results**: 
   - Processing type badge (single-digit or multi-digit)
   - Recognized digit(s) with confidence scores
   - Individual model predictions (expandable)

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list
- At least one CNN model file (svhn_model.h5 or usps_model.h5)
- Optional: OCR engines for enhanced recognition

## Notes

- EMNIST model is loaded but excluded from ensemble predictions
- Styles are compiled automatically when using `start-app.sh` or `start-app.bat`
- The app gracefully handles missing optional dependencies (OCR engines, YOLO)

## Troubleshooting

- **Styles not compiling**: Run `streamlit-sal compile` manually
- **Models not found**: Ensure model files are in the `models/` directory
- **Import errors**: Install dependencies with `pip install -r requirements.txt`

