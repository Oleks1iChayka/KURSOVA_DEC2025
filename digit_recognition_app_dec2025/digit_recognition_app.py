"""
Digit Recognition Streamlit App
A simple application for digit recognition using ensemble of CNN and OCR models.
"""
import streamlit as st
import streamlit_sal as sal
from streamlit_sal import sal_stylesheet

from components import render_app_header, render_digit_recognition_result
from constants import *
from utils import get_app_name
from PIL import Image

# Digit recognition imports
try:
    from digit_models_loader import load_all_models
    from digit_recognition import DigitRecognitionEngine
    DIGIT_RECOGNITION_AVAILABLE = True
except ImportError as e:
    DIGIT_RECOGNITION_AVAILABLE = False
    import logging
    logging.warning(f"Digit recognition not available: {e}")

# Basic application page configuration
st.set_page_config(
    page_title=get_app_name(),
    page_icon=APP_FAVICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=SIDEBAR_DEFAULT_STATE
)


def start_streamlit():
    """Main Streamlit app function"""
    # Wraps the application with a SAL stylesheet so elements within it can be customized
    with sal_stylesheet():
        render_app_header()

        st.markdown("---")
        
        # Instructions
        st.markdown("### 📤 Upload Images")
        st.markdown("Upload one or more images containing handwritten digits to recognize them.")

        # Image upload widget
        uploaded_images = st.file_uploader(
            I18N_IMAGE_UPLOAD_LABEL,
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            accept_multiple_files=True,
            help=I18N_IMAGE_UPLOAD_HELP,
            key="image_uploader"
        )
        
        # Convert uploaded images to PIL Images
        images = []
        if uploaded_images:
            for uploaded_file in uploaded_images:
                try:
                    img = Image.open(uploaded_file)
                    images.append(img)
                except Exception as e:
                    st.error(f"Error loading image {uploaded_file.name}: {e}")
        
        # Digit recognition section
        if images:
            st.markdown("---")
            st.markdown("### 🔢 Digit Recognition")
            
            if not DIGIT_RECOGNITION_AVAILABLE:
                st.error("Digit recognition is not available. Please check that all required packages are installed.")
            elif not DIGIT_RECOGNITION_ENABLED:
                st.warning("Digit recognition is disabled.")
            else:
                # Initialize digit recognition engine (cached)
                @st.cache_resource
                def get_digit_engine():
                    import os
                    models_dir = os.path.join(os.path.dirname(__file__), DIGIT_MODELS_DIR.lstrip('./'))
                    all_models = load_all_models(models_dir)
                    
                    if len(all_models['cnn_models']) == 0 and len(all_models['ocr_readers']) == 0:
                        return None
                    
                    return DigitRecognitionEngine(
                        models_dict=all_models['cnn_models'],
                        ocr_readers_dict=all_models['ocr_readers'],
                        yolo_model=all_models['yolo_model']
                    )
                
                engine = get_digit_engine()
                
                if engine is None:
                    st.warning(I18N_DIGIT_NO_MODELS)
                else:
                    # Process each image automatically
                    for idx, img in enumerate(images):
                        st.markdown(f"#### Image {idx + 1}")
                        with st.spinner(f"Recognizing digits in image {idx + 1}..."):
                            try:
                                result = engine.predict(img, auto_detect_multi=DIGIT_AUTO_DETECT)
                                
                                if result and 'error' not in result:
                                    # Display result
                                    render_digit_recognition_result(result, img)
                                else:
                                    st.error(f"Digit recognition failed for image {idx + 1}")
                            except Exception as e:
                                st.error(f"Error recognizing digits: {e}")
                        st.markdown("---")
        else:
            # Show empty state
            st.info("👆 Please upload an image above to get started with digit recognition.")


if __name__ == "__main__":
    start_streamlit()

