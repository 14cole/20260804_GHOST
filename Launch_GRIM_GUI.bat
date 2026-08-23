@echo off
setlocal
cd /d "%~dp0"

if not exist "GRIM_Revised_2\grim_cut_gui.py" (
    echo ERROR: GRIM_Revised_2\grim_cut_gui.py was not found.
    echo Keep the complete branch folder structure together.
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
py -3 -c "import sys; sys.path.insert(0, r'%~dp0GRIM_Revised_2'); import grim_cut_gui" >"%TEMP%\grim-gui-launch.log" 2>&1
if errorlevel 1 goto missing_dependencies
start "" pyw -3 "%~dp0GRIM_Revised_2\grim_cut_gui.py"
exit /b 0

:use_python
python -c "import sys; sys.path.insert(0, r'%~dp0GRIM_Revised_2'); import grim_cut_gui" >"%TEMP%\grim-gui-launch.log" 2>&1
if errorlevel 1 goto missing_dependencies
where pythonw.exe >nul 2>&1
if errorlevel 1 (
    python "%~dp0GRIM_Revised_2\grim_cut_gui.py"
) else (
    start "" pythonw "%~dp0GRIM_Revised_2\grim_cut_gui.py"
)
exit /b 0

:missing_dependencies
echo ERROR: The selected Python interpreter could not import GRIM.
echo.
if exist "%TEMP%\grim-gui-launch.log" type "%TEMP%\grim-gui-launch.log"
echo.
echo From this folder, install the application and dependencies with:
echo     py -3 -m pip install -e .
echo.
echo Then run this launcher again.
pause
exit /b 1
