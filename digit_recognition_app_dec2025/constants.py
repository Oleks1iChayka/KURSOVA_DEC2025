# Set asset path or remote url
APP_LOGO = './assets/seagull-logo.png'
APP_FAVICON = './assets/favicon.png'
# Set app layout to either 'centered' or 'wide'
APP_LAYOUT = 'centered'

# If you have additional information to show in a sidebar, you can enable it here
SHOW_SIDEBAR = False
# Set whether sidebar should be expanded at app load. ('collapsed'/'expanded')
SIDEBAR_DEFAULT_STATE = 'expanded'

# Translations
I18N_APP_NAME_DEFAULT = "Digit Recognition Streamlit App"  # Fallback name
I18N_APP_DESCRIPTION = "Digit recognition using ensemble of CNN and OCR models"

I18N_FORMAT_CONFIDENCE = "{}%"  # Place unit before or after {}
I18N_IMAGE_UPLOAD_LABEL = "Upload an image"
I18N_IMAGE_UPLOAD_HELP = "Upload one or more images containing handwritten digits"

# Digit Recognition
DIGIT_RECOGNITION_ENABLED = True
DIGIT_MODELS_DIR = "./models"  # Relative to src/
DIGIT_YOLO_MODEL_PATH = "yolov8n.pt"
DIGIT_AUTO_DETECT = False  # Auto-detect digits in images
I18N_DIGIT_RECOGNITION_LABEL = "Recognize digits in image"
I18N_DIGIT_RECOGNITION_HELP = "Enable digit recognition using ensemble of CNN and OCR models"
I18N_DIGIT_NO_MODELS = "⚠️ No digit recognition models available. Please ensure model files are in the models/ directory."
I18N_DIGIT_PREDICTED = "Predicted Digit"
I18N_DIGIT_MULTI_DIGIT = "Multi-Digit Sequence"
I18N_DIGIT_CONFIDENCE = "Confidence"
I18N_DIGIT_METHOD = "Method"


