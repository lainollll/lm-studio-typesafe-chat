@echo off
cd /d "%~dp0"
python -B chatbot.py
if errorlevel 1 pause
