"""
Model loader for digit recognition ensemble.
Handles graceful degradation if models or OCR engines are missing.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def load_cnn_models(models_dir: str) -> Dict[str, Any]:
    """
    Load CNN models (SVHN, EMNIST, USPS).
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dictionary of model_name -> model instance
    """
    models = {}
    model_files = {
        'SVHN': 'svhn_model.h5',
        'EMNIST': 'emnist_model.h5',
        'USPS': 'usps_model.h5'
    }
    
    for name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                models[name] = tf.keras.models.load_model(model_path)
                logger.info(f"✅ {name} model loaded from {model_path}")
            except ImportError:
                logger.warning(f"⚠️ TensorFlow not available - cannot load {name} model")
            except Exception as e:
                logger.warning(f"⚠️ Error loading {name} model: {e}")
        else:
            logger.warning(f"⚠️ {name} model not found: {model_path}")
    
    return models


def load_easyocr_reader() -> Optional[Any]:
    """
    Initialize EasyOCR reader (optional).
    
    Returns:
        EasyOCR reader instance or None if unavailable
    """
    try:
        import easyocr
        logger.info("Initializing EasyOCR (first time may download models)...")
        reader = easyocr.Reader(['en'], gpu=False)  # English only, CPU mode
        logger.info("✅ EasyOCR initialized")
        return reader
    except ImportError:
        logger.info("⚠️ EasyOCR not installed - skipping")
        return None
    except Exception as e:
        logger.warning(f"⚠️ EasyOCR initialization failed: {e}")
        return None


def load_tesseract_reader() -> Optional[bool]:
    """
    Check if Tesseract OCR is available (optional).
    
    Returns:
        True if available, None if not
    """
    try:
        import pytesseract
        # Test if tesseract is available
        version = pytesseract.get_tesseract_version()
        logger.info(f"✅ Tesseract version {version} found")
        return True
    except ImportError:
        logger.info("⚠️ pytesseract not installed - skipping Tesseract")
        return None
    except Exception as e:
        logger.info(f"⚠️ Tesseract not available: {e}")
        logger.info("   Install instructions:")
        logger.info("   - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("   - Mac: brew install tesseract")
        logger.info("   - Linux: sudo apt-get install tesseract-ocr")
        return None


def load_paddleocr_reader() -> Optional[Any]:
    """
    Initialize PaddleOCR reader (optional).
    
    Returns:
        PaddleOCR reader instance or None if unavailable
    """
    try:
        from paddleocr import PaddleOCR
        import logging as py_logging
        
        # Suppress PaddleOCR verbose logging
        try:
            from paddleocr import logger as paddle_logger
            paddle_logger.setLevel(py_logging.ERROR)
        except:
            pass
        
        logger.info("Initializing PaddleOCR (may download models on first run)...")
        
        # Try minimal initialization
        try:
            ocr = PaddleOCR(lang='en', use_angle_cls=False, show_log=False)
            logger.info("✅ PaddleOCR initialized")
            return ocr
        except Exception as e:
            logger.warning(f"⚠️ PaddleOCR initialization failed: {e}")
            return None
            
    except ImportError:
        logger.info("⚠️ PaddleOCR not installed - skipping")
        return None
    except Exception as e:
        logger.warning(f"⚠️ PaddleOCR error: {e}")
        return None


def load_yolo_model(model_path: str = "yolov8n.pt") -> Optional[Any]:
    """
    Load YOLO model for multi-digit detection (optional).
    
    Args:
        model_path: Path to YOLO model file
        
    Returns:
        YOLO model instance or None if unavailable
    """
    try:
        from ultralytics import YOLO
        
        # Check if model file exists
        if not os.path.exists(model_path):
            # Try in models directory
            alt_path = os.path.join(os.path.dirname(__file__), 'models', os.path.basename(model_path))
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                logger.warning(f"⚠️ YOLO model not found: {model_path}")
                logger.info("   YOLO will download model on first use if ultralytics is available")
        
        logger.info(f"Loading YOLO model: {model_path}")
        model = YOLO(model_path)
        logger.info("✅ YOLO model loaded")
        return model
    except ImportError:
        logger.info("⚠️ ultralytics not installed - skipping YOLO")
        return None
    except Exception as e:
        logger.warning(f"⚠️ YOLO model loading failed: {e}")
        return None


def load_all_models(models_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load all available models and OCR readers.
    
    Args:
        models_dir: Directory containing model files. If None, uses default location.
        
    Returns:
        Dictionary with keys:
        - 'cnn_models': Dict of CNN models
        - 'ocr_readers': Dict of OCR readers
        - 'yolo_model': YOLO model or None
    """
    if models_dir is None:
        # Default to models directory relative to this file
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    result = {
        'cnn_models': {},
        'ocr_readers': {},
        'yolo_model': None
    }
    
    # Load CNN models
    logger.info("Loading CNN models...")
    result['cnn_models'] = load_cnn_models(models_dir)
    
    # Load OCR readers
    logger.info("Loading OCR readers...")
    
    # EasyOCR
    easyocr_reader = load_easyocr_reader()
    if easyocr_reader is not None:
        result['ocr_readers']['easyocr'] = easyocr_reader
    
    # Tesseract (optional)
    tesseract_available = load_tesseract_reader()
    if tesseract_available:
        result['ocr_readers']['tesseract'] = True  # Use True as placeholder
    
    # PaddleOCR (optional)
    paddleocr_reader = load_paddleocr_reader()
    if paddleocr_reader is not None:
        result['ocr_readers']['paddleocr'] = paddleocr_reader
    
    # Load YOLO model
    logger.info("Loading YOLO model...")
    result['yolo_model'] = load_yolo_model()
    
    # Summary
    total_models = len(result['cnn_models']) + len(result['ocr_readers'])
    logger.info(f"📊 Total models loaded: {len(result['cnn_models'])} CNN + {len(result['ocr_readers'])} OCR = {total_models}")
    
    if total_models == 0:
        logger.warning("⚠️ No models available! Digit recognition will not work.")
        logger.info("   Please ensure:")
        logger.info("   1. Model files are in the models/ directory")
        logger.info("   2. Required packages are installed")
    
    return result

