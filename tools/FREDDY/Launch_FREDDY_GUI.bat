@echo off
setlocal
cd /d "%~dp0"

if not exist "impedance_gui.py" (
    echo ERROR: impedance_gui.py was not found.
    echo Keep the complete tools\FREDDY folder together.
    pause
    exit /b 1
)

where py.exe >nul 2>&1
if errorlevel 1 goto check_python
py -3 -V >nul 2>&1
if not errorlevel 1 goto use_py_launcher

:check_python
where python.exe >nul 2>&1
if errorlevel 1 goto no_python
python -V >nul 2>&1
if not errorlevel 1 goto use_python

:no_python
echo ERROR: Python 3 was not found.
echo Install Python 3.10 or newer, then run this launcher again.
pause
exit /b 1

:use_py_launcher
py -3 -c "import numpy, scipy; from ibc.ui import QT_AVAILABLE, MPL_AVAILABLE; assert QT_AVAILABLE, 'PySide6 is not available'; assert MPL_AVAILABLE, 'The Matplotlib Qt backend is not available'" >"%TEMP%\freddy-gui-launch.log" 2>&1
if errorlevel 1 goto missing_dependencies
where pyw.exe >nul 2>&1
if errorlevel 1 (
    start "" py.exe -3 "%~dp0impedance_gui.py"
) else (
    start "" pyw.exe -3 "%~dp0impedance_gui.py"
)
exit /b 0

:use_python
python -c "import numpy, scipy; from ibc.ui import QT_AVAILABLE, MPL_AVAILABLE; assert QT_AVAILABLE, 'PySide6 is not available'; assert MPL_AVAILABLE, 'The Matplotlib Qt backend is not available'" >"%TEMP%\freddy-gui-launch.log" 2>&1
if errorlevel 1 goto missing_dependencies
where pythonw.exe >nul 2>&1
if errorlevel 1 (
    start "" python.exe "%~dp0impedance_gui.py"
) else (
    start "" pythonw.exe "%~dp0impedance_gui.py"
)
exit /b 0

:missing_dependencies
echo ERROR: The selected Python interpreter could not import the FREDDY GUI.
echo.
if exist "%TEMP%\freddy-gui-launch.log" type "%TEMP%\freddy-gui-launch.log"
echo.
echo From this folder, install FREDDY's dependencies with:
echo     py -3 -m pip install -r requirements.txt
echo.
echo Then run this launcher again.
pause
exit /b 1
