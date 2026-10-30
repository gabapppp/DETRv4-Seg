@echo off
setlocal
cd /d "%~dp0"
py -3.12 -c "pass" >nul 2>&1
if not errorlevel 1 goto py312
py -3.13 -c "pass" >nul 2>&1
if not errorlevel 1 goto py313
py -3.11 -c "pass" >nul 2>&1
if not errorlevel 1 goto py311
python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,13)" >nul 2>&1
if not errorlevel 1 goto plain_python
echo Please install Python 3.12 from https://www.python.org/downloads/
pause
exit /b 1
:py312
py -3.12 run_viewer.py %*
goto done
:py313
py -3.13 run_viewer.py %*
goto done
:py311
py -3.11 run_viewer.py %*
goto done
:plain_python
python run_viewer.py %*
:done
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
