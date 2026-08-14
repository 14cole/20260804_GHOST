@echo off
setlocal
cd /d "%~dp0"

if not exist "Backend\ghost_gui.py" (
    echo ERROR: Backend\ghost_gui.py was not found.
    pause
    exit /b 1
)

where py.exe >nul 2>&1
if not errorlevel 1 goto use_py_launcher

where python.exe >nul 2>&1
if not errorlevel 1 goto use_python

echo ERROR: Python 3 was not found.
echo Install Python 3.10 or newer, then run this launcher again.
pause
exit /b 1

:use_py_launcher
py -3 "Backend\ghost_gui.py" --check >nul 2>&1
if errorlevel 1 goto missing_dependencies
start "" pyw -3 "%~dp0Backend\ghost_gui.py"
exit /b 0

:use_python
python "Backend\ghost_gui.py" --check >nul 2>&1
if errorlevel 1 goto missing_dependencies
where pythonw.exe >nul 2>&1
if errorlevel 1 (
    python "%~dp0Backend\ghost_gui.py"
) else (
    start "" pythonw "%~dp0Backend\ghost_gui.py"
)
exit /b 0

:missing_dependencies
echo ERROR: One or more GHOST GUI dependencies could not be imported.
echo.
echo Install them with:
echo     py -3 -m pip install numpy scipy matplotlib PySide6
echo.
echo Then run this launcher again.
pause
exit /b 1
