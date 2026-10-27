@echo off
rem Double-click to start Showing Planner; it opens in your browser (usually http://localhost:8501).
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run app/main.py
pause
