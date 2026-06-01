@echo off
REM RAG Gradio 前端启动脚本

set SCRIPT_DIR=%~dp0

call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
python "%SCRIPT_DIR%app.py"

pause
