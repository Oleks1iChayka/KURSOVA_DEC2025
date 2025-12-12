#!/usr/bin/env bash
echo "Starting Digit Recognition Streamlit App"

# ============================================
# OpenAI Configuration
# ============================================
# Set your OpenAI API key here (optional - app works in ML model recognition mode without it)
# Get your key from: https://platform.openai.com/api-keys
# Leave empty to use ML model recognition mode only
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# Optional: Model selection (default: gpt-4)
# Use gpt-4-vision-preview for image support
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4}"

# Optional: Enable streaming
export ENABLE_STREAMING="${ENABLE_STREAMING:-False}"

# Optional: System prompt
# export SYSTEM_PROMPT="You are a helpful assistant."

# Optional: App name
# export APP_NAME="Digit Recognition App"

# ============================================
# Compile Styles
# ============================================
echo "Compiling styles..."
streamlit-sal compile
if [ $? -ne 0 ]; then
    echo "Error: Failed to compile styles. Make sure streamlit-sal is installed."
    echo "Run: pip install streamlit-sal"
    exit 1
fi

# ============================================
# Run the App
# ============================================
echo ""
echo "Starting Streamlit app..."
echo ""
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Running in ML MODEL RECOGNITION MODE - no OpenAI API key configured"
    echo "Set OPENAI_API_KEY to enable OpenAI AI chat responses (optional)"
else
    echo "Using OpenAI API with model: $OPENAI_MODEL"
fi
echo ""
echo "The app will open in your default browser."
echo "If it doesn't, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run digit_recognition_app.py

