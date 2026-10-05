@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv-models\Scripts\python.exe" goto missing
".venv-models\Scripts\python.exe" -c "import torch,numpy,PIL" >nul 2>&1
if errorlevel 1 goto missing
".venv-models\Scripts\python.exe" app.py --open-try %*
set "GIRLPHOTO_EXIT=%ERRORLEVEL%"
echo Server stopped. If startup failed, read the message above.
pause
exit /b %GIRLPHOTO_EXIT%
:missing
echo First run setup_models.bat to install the isolated model environment.
pause
exit /b 1
