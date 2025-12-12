"""
Digit Recognition Engine using ensemble of CNN and OCR models.
Adapted from alternative_recognition.ipynb for Streamlit integration.
"""
import os
import tempfile
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image
import io

logger = logging.getLogger(__name__)


class DigitRecognitionEngine:
    """
    Main engine for digit recognition using ensemble methods.
    Combines CNN models (SVHN, EMNIST, USPS) with OCR models (EasyOCR, Tesseract, PaddleOCR).
    """
    
    def __init__(self, models_dict: Optional[Dict] = None, 
                 ocr_readers_dict: Optional[Dict] = None,
                 yolo_model: Optional[Any] = None):
        """
        Initialize the recognition engine.
        
        Args:
            models_dict: Dictionary of CNN models {'name': model}
            ocr_readers_dict: Dictionary of OCR readers {'name': reader}
            yolo_model: YOLO model instance for multi-digit detection
        """
        self.models_dict = models_dict or {}
        self.ocr_readers_dict = ocr_readers_dict or {}
        self.yolo_model = yolo_model
        
    def preprocess_for_model(self, image: Union[str, Image.Image, np.ndarray], 
                            target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        Preprocess image for CNN models (single digit).
        
        Args:
            image: Image path, PIL Image, or numpy array
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed image array shape (1, 28, 28, 1)
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # Handle RGBA
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        arr = np.array(img, dtype=np.float32) / 255.0
        
        # Invert if needed (MNIST format: white digit on black background)
        if np.mean(arr) > 0.5:
            arr = 1.0 - arr
        
        return arr.reshape(1, 28, 28, 1)
    
    def recognize_digit_easyocr(self, image_path: str) -> Tuple[Optional[int], float]:
        """Recognize digit using EasyOCR."""
        reader = self.ocr_readers_dict.get('easyocr')
        if reader is None:
            return None, 0.0
        
        try:
            results = reader.readtext(image_path)
            if results:
                text = results[0][1].strip()
                confidence = results[0][2]
                
                # Extract first digit
                for char in text:
                    if char.isdigit():
                        return int(char), float(confidence)
            return None, 0.0
        except Exception as e:
            logger.warning(f"EasyOCR error: {e}")
            return None, 0.0
    
    def recognize_digit_tesseract(self, image_path: str, preprocess: bool = True) -> Tuple[Optional[int], float]:
        """Recognize digit using Tesseract OCR."""
        if 'tesseract' not in self.ocr_readers_dict:
            return None, 0.0
        
        try:
            import pytesseract
            
            # Preprocess image
            if preprocess:
                img = self._preprocess_for_tesseract(image_path)
            else:
                img = Image.open(image_path) if isinstance(image_path, str) else image_path
            
            if img is None:
                return None, 0.0
            
            # Tesseract configuration for single digit
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
            
            # Get detailed data
            data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Find best result
            best_digit = None
            best_conf = 0.0
            
            for i, text in enumerate(data['text']):
                conf = float(data['conf'][i])
                if conf < 0 or not text.strip():
                    continue
                
                for char in text.strip():
                    if char.isdigit():
                        digit = int(char)
                        if conf > best_conf:
                            best_digit = digit
                            best_conf = conf
                        break
            
            # Fallback
            if best_digit is None:
                text = pytesseract.image_to_string(img, config=custom_config).strip()
                for char in text:
                    if char.isdigit():
                        best_digit = int(char)
                        best_conf = 50.0
                        break
            
            confidence = best_conf / 100.0 if best_conf > 0 else 0.0
            return best_digit, confidence
            
        except ImportError:
            return None, 0.0
        except Exception as e:
            logger.warning(f"Tesseract error: {e}")
            return None, 0.0
    
    def recognize_digit_paddleocr(self, image_path: str) -> Tuple[Optional[int], float]:
        """Recognize digit using PaddleOCR."""
        reader = self.ocr_readers_dict.get('paddleocr')
        if reader is None:
            return None, 0.0
        
        try:
            result = reader.ocr(image_path, cls=False)
            
            if not result or not result[0]:
                return None, 0.0
            
            # Parse result
            for line in result[0]:
                if len(line) >= 2 and len(line[1]) >= 2:
                    text = str(line[1][0]).strip()
                    confidence = float(line[1][1])
                    
                    # Extract first digit
                    for char in text:
                        if char.isdigit():
                            return int(char), confidence
            
            return None, 0.0
        except Exception as e:
            logger.warning(f"PaddleOCR error: {e}")
            return None, 0.0
    
    def _preprocess_for_tesseract(self, image_path: Union[str, Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """Preprocess image for Tesseract OCR."""
        try:
            import cv2
            
            # Read image
            if isinstance(image_path, str):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            elif isinstance(image_path, Image.Image):
                img = np.array(image_path.convert('L'))
            else:
                img = image_path
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if img is None:
                return None
            
            # Resize if too small
            height, width = img.shape
            if height < 50 or width < 50:
                scale = max(50 / height, 50 / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Thresholding
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Denoise
            img = cv2.medianBlur(img, 3)
            
            # Convert to PIL
            pil_img = Image.fromarray(img)
            
            # Enhance contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(2.0)
            
            return pil_img
            
        except Exception as e:
            logger.warning(f"Tesseract preprocessing error: {e}")
            try:
                if isinstance(image_path, str):
                    return Image.open(image_path)
                elif isinstance(image_path, Image.Image):
                    return image_path
                else:
                    return Image.fromarray(image_path)
            except:
                return None
    
    def detect_multiple_digits_yolo(self, image_path: str, min_digits: int = 2) -> Tuple[bool, int, Optional[Any]]:
        """Detect if image contains multiple digits using YOLO."""
        if self.yolo_model is None:
            return False, 0, None
        
        try:
            results = self.yolo_model(image_path)
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                digit_count = len(boxes)
                
                if digit_count >= min_digits:
                    return True, digit_count, boxes
                elif digit_count == 1:
                    return False, 1, boxes
                else:
                    return False, 0, None
            else:
                return False, 0, None
                
        except Exception as e:
            logger.warning(f"YOLO detection error: {e}")
            return False, 0, None
    
    def predict_single_digit(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Predict single digit using ensemble of all available models.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Dictionary with predictions from all models and ensemble result
        """
        results = {}
        
        # Save PIL Image to temp file if needed
        temp_file = None
        image_path = None
        
        try:
            if isinstance(image, Image.Image):
                # Save to temp file for OCR
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                image.save(temp_file.name)
                image_path = temp_file.name
            else:
                image_path = image
            
            # Preprocess for CNN models
            try:
                img_array = self.preprocess_for_model(image)
            except Exception as e:
                logger.error(f"Error preprocessing image: {e}")
                return {'error': str(e)}
            
            # CNN model predictions (exclude EMNIST)
            for model_name, model in self.models_dict.items():
                # Skip EMNIST model - keep files and code but don't use in predictions
                if model_name == 'EMNIST':
                    logger.info(f"Skipping {model_name} model (excluded from ensemble)")
                    continue
                    
                try:
                    if model is None:
                        continue
                    pred = model.predict(img_array, verbose=0)[0]
                    digit = int(np.argmax(pred))
                    confidence = float(pred[digit])
                    results[model_name] = {
                        'digit': digit,
                        'confidence': confidence,
                        'probabilities': pred.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Error with {model_name}: {e}")
                    continue
            
            # OCR predictions
            if 'easyocr' in self.ocr_readers_dict:
                digit, confidence = self.recognize_digit_easyocr(image_path)
                if digit is not None:
                    results['easyocr'] = {
                        'digit': digit,
                        'confidence': confidence,
                        'probabilities': None
                    }
            
            if 'tesseract' in self.ocr_readers_dict:
                digit, confidence = self.recognize_digit_tesseract(image_path, preprocess=True)
                if digit is not None:
                    results['tesseract'] = {
                        'digit': digit,
                        'confidence': confidence,
                        'probabilities': None
                    }
            
            if 'paddleocr' in self.ocr_readers_dict:
                digit, confidence = self.recognize_digit_paddleocr(image_path)
                if digit is not None:
                    results['paddleocr'] = {
                        'digit': digit,
                        'confidence': confidence,
                        'probabilities': None
                    }
            
            # Ensemble: Weighted voting
            if results:
                votes = {i: 0.0 for i in range(10)}
                
                for model_name, result in results.items():
                    digit = result['digit']
                    confidence = result['confidence']
                    votes[digit] += confidence
                
                ensemble_digit = max(votes, key=votes.get)
                ensemble_confidence = votes[ensemble_digit] / len(results) if len(results) > 0 else 0.0
                
                results['ensemble'] = {
                    'digit': ensemble_digit,
                    'confidence': ensemble_confidence,
                    'votes': votes,
                    'num_models': len(results) - 1  # Exclude ensemble itself
                }
            else:
                results['ensemble'] = {
                    'digit': None,
                    'confidence': 0.0,
                    'votes': {i: 0.0 for i in range(10)},
                    'num_models': 0
                }
            
            return results
            
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    def predict_multi_digit(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Predict multiple digits using YOLO detection + OCR recognition.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Dictionary with sequence, confidence, method, etc.
        """
        # Save PIL Image to temp file if needed
        temp_file = None
        image_path = None
        
        try:
            if isinstance(image, Image.Image):
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                image.save(temp_file.name)
                image_path = temp_file.name
            else:
                image_path = image
            
            results = {
                'sequence': [],
                'confidence': [],
                'full_text': None,
                'method': None
            }
            
            # Method 1: YOLO detection + OCR recognition
            is_multi, digit_count, boxes = self.detect_multiple_digits_yolo(image_path, min_digits=2)
            
            if boxes is not None and digit_count >= 2:
                # Try OCR to recognize all digits
                ocr_text = None
                ocr_confidence = None
                
                # Try EasyOCR
                if 'easyocr' in self.ocr_readers_dict:
                    try:
                        ocr_results = self.ocr_readers_dict['easyocr'].readtext(image_path)
                        if ocr_results:
                            full_text = ''.join([result[1] for result in ocr_results])
                            digits = [c for c in full_text if c.isdigit()]
                            if digits:
                                ocr_text = ''.join(digits)
                                confidences = [result[2] for result in ocr_results]
                                ocr_confidence = np.mean(confidences) if confidences else 0.5
                                results['method'] = 'yolo_easyocr'
                    except Exception as e:
                        logger.warning(f"EasyOCR multi-digit error: {e}")
                
                # Try Tesseract
                if ocr_text is None and 'tesseract' in self.ocr_readers_dict:
                    try:
                        import pytesseract
                        img = Image.open(image_path)
                        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
                        text = pytesseract.image_to_string(img, config=config).strip()
                        digits = [c for c in text if c.isdigit()]
                        if digits:
                            ocr_text = ''.join(digits)
                            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                            confidences = [float(c) for c in data['conf'] if float(c) > 0]
                            ocr_confidence = np.mean(confidences) / 100.0 if confidences else 0.5
                            results['method'] = 'yolo_tesseract'
                    except Exception as e:
                        logger.warning(f"Tesseract multi-digit error: {e}")
                
                # Try PaddleOCR
                if ocr_text is None and 'paddleocr' in self.ocr_readers_dict:
                    try:
                        ocr_results = self.ocr_readers_dict['paddleocr'].ocr(image_path, cls=False)
                        if ocr_results and ocr_results[0]:
                            full_text = ''.join([line[1][0] for line in ocr_results[0]])
                            digits = [c for c in full_text if c.isdigit()]
                            if digits:
                                ocr_text = ''.join(digits)
                                confidences = [line[1][1] for line in ocr_results[0]]
                                ocr_confidence = np.mean(confidences) if confidences else 0.5
                                results['method'] = 'yolo_paddleocr'
                    except Exception as e:
                        logger.warning(f"PaddleOCR multi-digit error: {e}")
                
                if ocr_text:
                    results['full_text'] = ocr_text
                    results['sequence'] = [int(d) for d in ocr_text]
                    results['confidence'] = [ocr_confidence] * len(ocr_text)
                    return results
            
            # Method 2: OCR-only fallback (if YOLO not available)
            for ocr_name in ['easyocr', 'tesseract', 'paddleocr']:
                if ocr_name not in self.ocr_readers_dict:
                    continue
                
                try:
                    if ocr_name == 'easyocr':
                        ocr_results = self.ocr_readers_dict['easyocr'].readtext(image_path)
                        if ocr_results:
                            full_text = ''.join([result[1] for result in ocr_results])
                            digits = [c for c in full_text if c.isdigit()]
                            if digits and len(digits) >= 2:
                                ocr_text = ''.join(digits)
                                confidences = [result[2] for result in ocr_results]
                                ocr_confidence = np.mean(confidences) if confidences else 0.5
                                results['method'] = 'ocr_easyocr'
                                break
                    
                    elif ocr_name == 'tesseract':
                        import pytesseract
                        img = Image.open(image_path)
                        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
                        text = pytesseract.image_to_string(img, config=config).strip()
                        digits = [c for c in text if c.isdigit()]
                        if digits and len(digits) >= 2:
                            ocr_text = ''.join(digits)
                            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                            confidences = [float(c) for c in data['conf'] if float(c) > 0]
                            ocr_confidence = np.mean(confidences) / 100.0 if confidences else 0.5
                            results['method'] = 'ocr_tesseract'
                            break
                    
                    elif ocr_name == 'paddleocr':
                        ocr_results = self.ocr_readers_dict['paddleocr'].ocr(image_path, cls=False)
                        if ocr_results and ocr_results[0]:
                            full_text = ''.join([line[1][0] for line in ocr_results[0]])
                            digits = [c for c in full_text if c.isdigit()]
                            if digits and len(digits) >= 2:
                                ocr_text = ''.join(digits)
                                confidences = [line[1][1] for line in ocr_results[0]]
                                ocr_confidence = np.mean(confidences) if confidences else 0.5
                                results['method'] = 'ocr_paddleocr'
                                break
                except Exception as e:
                    logger.warning(f"OCR multi-digit error ({ocr_name}): {e}")
                    continue
            
            if ocr_text:
                results['full_text'] = ocr_text
                results['sequence'] = [int(d) for d in ocr_text]
                results['confidence'] = [ocr_confidence] * len(ocr_text)
                return results
            
            return None
            
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    def predict(self, image: Union[str, Image.Image], auto_detect_multi: bool = True) -> Dict[str, Any]:
        """
        Main prediction function - automatically handles single or multi-digit.
        
        Args:
            image: Image path or PIL Image
            auto_detect_multi: If True, automatically detect multi-digit images
            
        Returns:
            Dictionary with predictions
        """
        # Auto-detect multi-digit images
        if auto_detect_multi:
            try:
                # Save to temp file for YOLO if needed
                temp_file = None
                image_path = None
                
                if isinstance(image, Image.Image):
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    image.save(temp_file.name)
                    image_path = temp_file.name
                else:
                    image_path = image
                
                is_multi, digit_count, boxes = self.detect_multiple_digits_yolo(image_path, min_digits=2)
                
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                
                if is_multi:
                    logger.info(f"Detected {digit_count} digit(s) - using multi-digit processing")
                    multi_result = self.predict_multi_digit(image)
                    if multi_result:
                        return {
                            'multi_digit': True,
                            'processing_type': 'multi-digit',
                            'sequence': multi_result['sequence'],
                            'full_text': multi_result['full_text'],
                            'confidence': multi_result['confidence'],
                            'method': multi_result['method'],
                            'ensemble': {
                                'digit': None,
                                'confidence': np.mean(multi_result['confidence']) if multi_result['confidence'] else 0.0,
                                'sequence': multi_result['sequence'],
                                'full_text': multi_result['full_text']
                            }
                        }
                    else:
                        logger.warning("Multi-digit processing failed, falling back to single-digit")
            except Exception as e:
                logger.warning(f"Multi-digit detection error: {e}, using single-digit processing")
        
        # Single digit processing
        single_result = self.predict_single_digit(image)
        single_result['multi_digit'] = False
        single_result['processing_type'] = 'single-digit'
        return single_result

