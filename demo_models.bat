@echo off
setlocal
cd /d "%~dp0"
call models.bat demo --out ml_runs\demo
set "GIRLPHOTO_EXIT=%ERRORLEVEL%"
pause
exit /b %GIRLPHOTO_EXIT%
