@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py %*
) else (
  python -c "import PIL" >nul 2>&1
  if errorlevel 1 (
    echo 首次使用请先双击 setup.bat 安装环境。
    pause
    exit /b 1
  )
  python app.py %*
)
set "GIRLPHOTO_EXIT=%ERRORLEVEL%"
echo.
echo 程序已停止。若启动失败，请查看上方的错误信息。
pause
exit /b %GIRLPHOTO_EXIT%
