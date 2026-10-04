@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
) else (
  python -c "import PIL" >nul 2>&1
  if errorlevel 1 (
    echo 首次使用请先双击 setup.bat 安装环境。
    pause
    exit /b 1
  )
  python app.py
)
if errorlevel 1 pause
