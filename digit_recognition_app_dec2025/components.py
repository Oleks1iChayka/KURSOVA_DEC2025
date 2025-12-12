import streamlit as st
import streamlit_sal as sal
from PIL import Image

from constants import (
    APP_LOGO,
    I18N_APP_DESCRIPTION,
    I18N_FORMAT_CONFIDENCE,
)
from utils import get_app_name


def render_app_header():
    app_name = get_app_name()
    app_description = I18N_APP_DESCRIPTION

    # Display logo with custom size (bigger than default)
    col_logo, col_content = st.columns([0.15, 0.85])
    with col_logo:
        st.image(APP_LOGO, width=120)  # Adjust width to make logo bigger (default is ~80px)

    # Remove flex grow so header does not take half the app height
    with col_content:
        with sal.columns('no-flex-grow'):
            col0 = st.columns([1])[0]
            with col0:
                with sal.subheader("app-header", container=col0):
                    col0.subheader(app_name, anchor=False)
                if app_description:
                    col0.caption(app_description)




def render_digit_recognition_result(result: dict, image: Image.Image):
    """
    Display digit recognition results in Streamlit.
    
    Args:
        result: Dictionary with predictions, confidence, etc.
        image: PIL Image that was recognized
    """
    from constants import (
        I18N_DIGIT_PREDICTED, I18N_DIGIT_MULTI_DIGIT, I18N_DIGIT_CONFIDENCE,
        I18N_DIGIT_METHOD, I18N_FORMAT_CONFIDENCE
    )
    
    # Show original image
    st.image(image, caption="Uploaded image", use_container_width=True)
    
    # Display processing type badge (less prominent)
    processing_type = result.get('processing_type', 'single-digit')
    is_multi = result.get('multi_digit', False)
    
    # Create a subtle badge showing processing type
    if is_multi or processing_type == 'multi-digit':
        st.markdown("""
        <div style="background-color: #e8f4f8; color: #1f77b4; padding: 8px; border-radius: 5px; margin: 10px 0; text-align: center; font-weight: normal; font-size: 14px; border: 1px solid #b3d9e6;">
            🔢 Processed as: Multi-digit image
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #e8f8e8; color: #2ca02c; padding: 8px; border-radius: 5px; margin: 10px 0; text-align: center; font-weight: normal; font-size: 14px; border: 1px solid #b3e6b3;">
            🔢 Processed as: Single-digit image
        </div>
        """, unsafe_allow_html=True)
    
    # Handle multi-digit results
    if result.get('multi_digit', False):
        st.subheader(f"🔢 {I18N_DIGIT_MULTI_DIGIT}")
        
        full_text = result.get('full_text', 'N/A')
        sequence = result.get('sequence', [])
        confidences = result.get('confidence', [])
        method = result.get('method', 'unknown')
        
        # Display full sequence
        st.markdown(f"**Sequence:** `{full_text}`")
        st.markdown(f"**{I18N_DIGIT_METHOD}:** {method}")
        
        # Show individual digits
        if sequence and confidences:
            st.markdown("**Individual Digits:**")
            cols = st.columns(len(sequence))
            for idx, (digit, conf) in enumerate(zip(sequence, confidences)):
                with cols[idx]:
                    st.metric(f"Digit {idx + 1}", digit, 
                             delta=f"{conf*100:.1f}%" if isinstance(conf, (int, float)) else "N/A")
        
        # Average confidence
        if confidences:
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            st.markdown(f"**Average {I18N_DIGIT_CONFIDENCE}:** {I18N_FORMAT_CONFIDENCE.format(avg_conf*100)}")
    
    # Handle single-digit results
    else:
        ensemble = result.get('ensemble', {})
        digit = ensemble.get('digit')
        conf = ensemble.get('confidence', 0.0)
        
        if digit is not None:
            st.subheader(f"🔢 {I18N_DIGIT_PREDICTED}: {digit}")
            st.markdown(f"**{I18N_DIGIT_CONFIDENCE}:** {I18N_FORMAT_CONFIDENCE.format(conf*100)}")
            
            # Show individual model predictions in expander
            with st.expander("**View individual model predictions**"):
                for model_name, pred in result.items():
                    # Skip non-dictionary entries (like 'multi_digit', 'processing_type', 'ensemble')
                    if model_name == 'ensemble' or not isinstance(pred, dict):
                        continue
                    pred_digit = pred.get('digit', 'N/A')
                    pred_conf = pred.get('confidence', 0.0)
                    st.write(f"**{model_name}:** {pred_digit} ({I18N_FORMAT_CONFIDENCE.format(pred_conf*100)})")
            
            # Show ensemble info
            num_models = ensemble.get('num_models', 0)
            if num_models > 0:
                st.caption(f"Based on {num_models} model(s)")
        else:
            st.warning("Could not recognize digit. Please try a different image.")

