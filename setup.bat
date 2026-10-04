@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo 安装完成。双击 start.bat 开始使用。
pause
exit /b 0
:failed
echo 安装失败。请检查 Python 3.10 或以上版本，以及安装依赖所需的网络连接。
pause
exit /b 1
