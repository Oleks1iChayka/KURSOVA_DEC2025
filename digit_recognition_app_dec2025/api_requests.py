"""
API request handler for ML model recognition mode or optional OpenAI API.
ML model recognition mode uses digit recognition models - no OpenAI required.
"""
import os
import time
import logging
import streamlit as st
from typing import Optional, Any

# OpenAI is optional - only import if available
try:
    from openai import OpenAI, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    APIError = Exception

from constants import STATUS_ERROR, STATUS_COMPLETED, DEFAULT_CHAT_MODEL_NAME, DEFAULT_VISION_MODEL, DEFAULT_VISION_MODEL
from utils import (
    sanitize_messages_for_request,
    set_result_message_state,
    ResponseProcessingError,
    handle_chat_api_error
)


def get_openai_client() -> Optional[Any]:
    """Get OpenAI client if API key is configured and OpenAI is available"""
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = st.session_state.get('openai_api_key')
    if not api_key:
        return None
    
    try:
        # Use OpenAI API directly
        return OpenAI(api_key=api_key)
    except Exception as e:
        logging.warning(f"OpenAI client initialization failed: {e}")
        return None


def get_ml_model_response(messages: list, has_images: bool = False) -> str:
    """Generate response using ML model recognition mode (no OpenAI required)"""
    user_messages = [msg for msg in messages if msg.get('role') == 'user']
    if not user_messages:
        return "Hello! How can I help you today?"
    
    last_user_msg = user_messages[-1]
    content = last_user_msg.get('content', '')
    
    # Handle multimodal content
    if isinstance(content, list):
        text_parts = [item.get('text', '') for item in content if item.get('type') == 'text']
        prompt = ' '.join(text_parts) if text_parts else ''
        has_images = any(item.get('type') == 'image_url' for item in content)
    else:
        prompt = content
        has_images = False
    
    if has_images:
        return f"I can see the image(s) you've shared. Based on the image{'s' if prompt else ''}, {'and your question: ' + prompt if prompt else 'I can help you analyze it.'} This app uses ML model recognition mode. Upload images with digits and use the 'Recognize digits' feature for digit recognition. To enable OpenAI AI responses, configure your OpenAI API key (optional)."
    
    # ML model recognition mode responses
    prompt_lower = prompt.lower()
    if 'hello' in prompt_lower or 'hi' in prompt_lower:
        return "Hello! This app uses ML model recognition mode for digit recognition. Upload images with digits and check 'Recognize digits' to use the ensemble models. To enable OpenAI AI chat responses, configure your OpenAI API key (optional)."
    elif 'help' in prompt_lower:
        return "I'm here to help! This app focuses on digit recognition using ML models. Upload images with digits and use the recognition feature. To enable OpenAI AI chat responses, set your OPENAI_API_KEY environment variable (optional)."
    elif '?' in prompt:
        return f"That's an interesting question about '{prompt[:50]}...'. This app uses ML model recognition mode. For digit recognition, upload images and use the recognition feature. For AI chat responses, configure your OpenAI API key (optional)."
    else:
        return f"I understand you're asking about: {prompt[:100]}{'...' if len(prompt) > 100 else ''}. This app uses ML model recognition mode for digit recognition. Upload images with digits to use the ensemble models. To enable OpenAI AI chat responses, configure your OpenAI API key (optional)."


def send_chat_api_request(message):
    """Send request to OpenAI Chat API or use ML model recognition mode"""
    meta_id = message['meta_id']
    openai_client = get_openai_client()
    
    # Check if we have images in the message
    has_images = False
    if isinstance(message.get('content'), list):
        has_images = any(item.get('type') == 'image_url' for item in message['content'])
    
    # Use ML model recognition mode if OpenAI not available or no API key
    if not openai_client:
        ml_response = get_ml_model_response(st.session_state.messages, has_images)
        set_result_message_state(
            meta_id, 
            ml_response, 
            status=STATUS_COMPLETED,
            citations=None,
            extra_model_output=None
        )
        return
    
    # Use real OpenAI API
    processed_citations = None
    extra_model_output = None
    
    with handle_chat_api_error(meta_id):
        if not OPENAI_AVAILABLE:
            raise Exception("OpenAI package not installed. Install with: pip install openai")
        
        model = st.session_state.get('openai_model', DEFAULT_CHAT_MODEL_NAME)
        if not model or model == DEFAULT_CHAT_MODEL_NAME:
            # Default to vision model for images, standard model for text
            model = DEFAULT_VISION_MODEL if has_images else DEFAULT_CHAT_MODEL_NAME
        
        completion = openai_client.chat.completions.create(
            model=model,
            messages=sanitize_messages_for_request(st.session_state.messages)
        )
        
        content = completion.choices[0].message.content
        
        # Extract usage info if available
        if hasattr(completion, 'usage'):
            extra_model_output = {
                'datarobot_token_count': completion.usage.total_tokens if completion.usage else None
            }
        
        set_result_message_state(
            meta_id, 
            content, 
            status=STATUS_COMPLETED,
            citations=processed_citations,
            extra_model_output=extra_model_output
        )
        return


def send_chat_api_streaming_request(message):
    """Send streaming request to OpenAI Chat API or use ML model recognition mode"""
    meta_id = message['meta_id']
    openai_client = get_openai_client()
    
    # Check if we have images in the message
    has_images = False
    if isinstance(message.get('content'), list):
        has_images = any(item.get('type') == 'image_url' for item in message['content'])
    
    # Use ML model recognition mode if OpenAI not available or no API key
    if not openai_client:
        ml_response = get_ml_model_response(st.session_state.messages, has_images)
        # Simulate streaming by yielding words
        words = ml_response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.05)  # Small delay to simulate streaming
        
        set_result_message_state(
            meta_id,
            ml_response,
            status=STATUS_COMPLETED,
            citations=None,
            extra_model_output=None
        )
        return
    
    # Use real OpenAI API with streaming
    processed_citations = None
    extra_model_output = None
    aggregated_content = ""
    
    with handle_chat_api_error(meta_id):
        if not OPENAI_AVAILABLE:
            raise Exception("OpenAI package not installed. Install with: pip install openai")
        
        model = st.session_state.get('openai_model', DEFAULT_CHAT_MODEL_NAME)
        if not model or model == DEFAULT_CHAT_MODEL_NAME:
            # Default to vision model for images, standard model for text
            model = DEFAULT_VISION_MODEL if has_images else DEFAULT_CHAT_MODEL_NAME
        
        streaming_response = openai_client.chat.completions.create(
            model=model,
            messages=sanitize_messages_for_request(st.session_state.messages),
            stream=True
        )
        
        for chunk in streaming_response:
            if len(chunk.choices) == 0:
                continue
            
            content = chunk.choices[0].delta.content
            is_final_chunk = chunk.choices[0].finish_reason == 'stop'
            aggregated_content += content if content is not None else ''
            
            if not is_final_chunk and content is not None:
                yield content
            elif is_final_chunk:
                # Extract usage info if available
                if hasattr(chunk, 'usage'):
                    extra_model_output = {
                        'datarobot_token_count': chunk.usage.total_tokens if chunk.usage else None
                    }
                
                set_result_message_state(
                    meta_id,
                    aggregated_content,
                    status=STATUS_COMPLETED,
                    citations=processed_citations,
                    extra_model_output=extra_model_output
                )
                return


def get_application_info():
    """Return empty dict for standalone version"""
    return {}

