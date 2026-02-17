@echo off
REM Starts Streamlit app in background (Windows)
start "StreamlitApp" /b python -m streamlit run app.py --server.port=8989 > logs.txt 2>&1
echo Streamlit started, logs -> logs.txt
