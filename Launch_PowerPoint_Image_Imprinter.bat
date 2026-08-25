@echo off
setlocal
cd /d "%~dp0"

if not exist "GRIM_Revised_2\ppt_image_imprinter.py" (
    echo ERROR: GRIM_Revised_2\ppt_image_imprinter.py was not found.
    pause
    exit /b 1
)

set "PPT_LAUNCH_LOG=%TEMP%\ppt-image-imprinter-launch.log"
type nul >"%PPT_LAUNCH_LOG%"

if exist "%~dp0.venv\Scripts\python.exe" (
    call :try_python "%~dp0.venv\Scripts\python.exe"
    if not errorlevel 1 goto launch_python
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    call :try_python "%VIRTUAL_ENV%\Scripts\python.exe"
    if not errorlevel 1 goto launch_python
)

where py.exe >nul 2>&1
if not errorlevel 1 (
    echo --- py.exe -3 --- >>"%PPT_LAUNCH_LOG%"
    py.exe -3 -c "import PySide6, pythoncom, win32com.client" >>"%PPT_LAUNCH_LOG%" 2>&1
    if not errorlevel 1 goto launch_py
)

where python.exe >nul 2>&1
if not errorlevel 1 (
    call :try_python "python.exe"
    if not errorlevel 1 goto launch_python
)

goto missing_dependencies

:try_python
set "PPT_PYTHON=%~1"
echo --- %PPT_PYTHON% --- >>"%PPT_LAUNCH_LOG%"
"%PPT_PYTHON%" -c "import PySide6, pythoncom, win32com.client" >>"%PPT_LAUNCH_LOG%" 2>&1
exit /b %ERRORLEVEL%

:launch_python
for %%I in ("%PPT_PYTHON%") do set "PPT_PYTHONW=%%~dpIpythonw.exe"
if exist "%PPT_PYTHONW%" (
    start "" "%PPT_PYTHONW%" "%~dp0GRIM_Revised_2\ppt_image_imprinter.py"
) else (
    start "" "%PPT_PYTHON%" "%~dp0GRIM_Revised_2\ppt_image_imprinter.py"
)
exit /b 0

:launch_py
where pyw.exe >nul 2>&1
if errorlevel 1 (
    start "" py.exe -3 "%~dp0GRIM_Revised_2\ppt_image_imprinter.py"
) else (
    start "" pyw.exe -3 "%~dp0GRIM_Revised_2\ppt_image_imprinter.py"
)
exit /b 0

:missing_dependencies
echo ERROR: No preferred Python interpreter has PySide6 and pywin32.
echo.
if exist "%PPT_LAUNCH_LOG%" type "%PPT_LAUNCH_LOG%"
echo.
echo Create the shared environment at %~dp0.venv, then install with:
echo     .venv\Scripts\python.exe -m pip install -e ".[powerpoint]"
echo.
pause
exit /b 1
