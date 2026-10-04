@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv-models\Scripts\python.exe" goto environment_ready
python -m venv .venv-models
if errorlevel 1 goto failed
:environment_ready
".venv-models\Scripts\python.exe" -c "from pathlib import Path; import sys,site; expected=Path('.venv-models').resolve(); config=(expected/'pyvenv.cfg').read_text(); assert Path(sys.prefix).resolve()==expected and sys.prefix!=sys.base_prefix, 'Not the project virtual environment'; assert 'include-system-site-packages = false' in config.lower() and site.ENABLE_USER_SITE is False, 'Environment is not isolated'; print('Using isolated environment:',sys.executable)"
if errorlevel 1 goto invalid_environment
if /i "%~1"=="--check" exit /b 0
if not "%~1"=="" set "HTTPS_PROXY=%~1"
".venv-models\Scripts\python.exe" -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv-models\Scripts\python.exe" -m pip install -r requirements-models.txt
if errorlevel 1 goto failed
".venv-models\Scripts\python.exe" -c "import sys,torch; assert sys.prefix != sys.base_prefix; print('Ready:',sys.executable); print('PyTorch:',torch.__version__)"
if errorlevel 1 goto failed
echo Ready. Run demo_models.bat to test both models.
pause
exit /b 0
:invalid_environment
echo The existing environment is invalid or shares system packages.
echo Stopped before installing anything. Do not install into your main Python.
pause
exit /b 1
:failed
echo Installation failed. Your main Python environment was not changed.
echo With a local HTTP proxy, try: setup_models.bat http://127.0.0.1:10792
pause
exit /b 1
