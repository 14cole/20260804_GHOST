@echo off
setlocal
cd /d "%~dp0"

if not exist "GRIM_Revised_2\ppt_image_imprinter.py" (
    echo ERROR: GRIM_Revised_2\ppt_image_imprinter.py was not found.
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
py -3 -c "import PySide6, pythoncom, win32com.client" >nul 2>&1
if errorlevel 1 goto missing_dependencies
start "" pyw -3 "%~dp0GRIM_Revised_2\ppt_image_imprinter.py"
exit /b 0

:use_python
python -c "import PySide6, pythoncom, win32com.client" >nul 2>&1
if errorlevel 1 goto missing_dependencies
where pythonw.exe >nul 2>&1
if errorlevel 1 (
    python "%~dp0GRIM_Revised_2\ppt_image_imprinter.py"
) else (
    start "" pythonw "%~dp0GRIM_Revised_2\ppt_image_imprinter.py"
)
exit /b 0

:missing_dependencies
echo ERROR: PySide6 and/or pywin32 is missing from this Python installation.
echo.
echo From this folder, run:
echo     py -3 -m pip install -e ".[powerpoint]"
echo.
pause
exit /b 1
