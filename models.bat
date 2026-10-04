@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv-models\Scripts\python.exe" goto missing
".venv-models\Scripts\python.exe" -c "import torch,numpy,PIL" >nul 2>&1
if errorlevel 1 goto missing
".venv-models\Scripts\python.exe" -m girlphoto_ml %*
set "GIRLPHOTO_EXIT=%ERRORLEVEL%"
if not "%GIRLPHOTO_EXIT%"=="0" pause
exit /b %GIRLPHOTO_EXIT%
:missing
echo First run setup_models.bat. Models use their own .venv-models environment.
pause
exit /b 1
