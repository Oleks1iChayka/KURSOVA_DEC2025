@echo off
echo Starting Digit Recognition Streamlit App

REM ============================================
REM OpenAI Configuration
REM ============================================
REM Set your OpenAI API key here (optional - app works in ML model recognition mode without it)
REM Get your key from: https://platform.openai.com/api-keys
REM Leave empty to use ML model recognition mode only
set OPENAI_API_KEY=

REM Optional: Model selection (default: gpt-4)
REM Use gpt-4-vision-preview for image support
set OPENAI_MODEL=gpt-4

REM Optional: Enable streaming
set ENABLE_STREAMING=False

REM Optional: System prompt
REM set SYSTEM_PROMPT=You are a helpful assistant.

REM Optional: App name
REM set APP_NAME=Digit Recognition App

REM ============================================
REM Digit Recognition Configuration
REM ============================================
REM Enable/disable digit recognition (default: True)
set DIGIT_RECOGNITION_ENABLED=True

REM ============================================
REM Compile Styles
REM ============================================
echo Compiling styles...
streamlit-sal compile
if errorlevel 1 (
    echo Error: Failed to compile styles. Make sure streamlit-sal is installed.
    echo Run: pip install streamlit-sal
    pause
    exit /b 1
)

REM ============================================
REM Run the App
REM ============================================
echo.
echo Starting Streamlit app...
echo.
if "%OPENAI_API_KEY%"=="" (
    echo Running in ML MODEL RECOGNITION MODE - no OpenAI API key configured
    echo Set OPENAI_API_KEY to enable OpenAI AI chat responses (optional)
) else (
    echo Using OpenAI API with model: %OPENAI_MODEL%
)
echo.
echo The app will open in your default browser.
echo If it doesn't, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

streamlit run digit_recognition_app.py

pause

