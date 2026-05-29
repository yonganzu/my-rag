@echo off
REM RAG 学习项目启动脚本
REM 使用虚拟环境中的 Python 运行主程序

REM 获取脚本所在目录
set SCRIPT_DIR=%~dp0

REM 激活虚拟环境并运行 main.py
call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
python "%SCRIPT_DIR%main.py"

REM 暂停以便查看输出
pause
