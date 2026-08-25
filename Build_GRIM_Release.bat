@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
  set "GRIM_RELEASE_PYTHON=%~dp0.venv\Scripts\python.exe"
  goto run_python
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
  set "GRIM_RELEASE_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
  goto run_python
)

where py.exe >nul 2>&1
if not errorlevel 1 (
  py.exe -3 -c "import sys" >nul 2>&1
  if not errorlevel 1 goto run_py
)

where python.exe >nul 2>&1
if not errorlevel 1 (
  set "GRIM_RELEASE_PYTHON=python.exe"
  goto run_python
)

echo Python 3 was not found. Install Python 3.10 or newer, then run this file again.
pause
exit /b 1

:run_python
"%GRIM_RELEASE_PYTHON%" "%~dp0build_release.py" %*
goto finished

:run_py
py.exe -3 "%~dp0build_release.py" %*

:finished
set "GRIM_RELEASE_EXIT=%ERRORLEVEL%"
if not "%GRIM_RELEASE_EXIT%"=="0" (
  echo.
  echo No existing release files were overwritten.
)
echo.
pause
exit /b %GRIM_RELEASE_EXIT%
