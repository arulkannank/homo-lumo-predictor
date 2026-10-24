@echo off
rem Starts the HOMO/LUMO predictor; it opens in the browser at http://localhost:8501
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run app.py
