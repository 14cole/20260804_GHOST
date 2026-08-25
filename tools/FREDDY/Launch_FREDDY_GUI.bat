@echo off
setlocal
cd /d "%~dp0"

if not exist "impedance_gui.py" (
    echo ERROR: impedance_gui.py was not found.
    echo Keep the complete tools\FREDDY folder together.
    pause
    exit /b 1
)

for %%I in ("%~dp0..\..") do set "GRIM_REPO_ROOT=%%~fI"
set "FREDDY_LAUNCH_LOG=%TEMP%\freddy-gui-launch.log"
type nul >"%FREDDY_LAUNCH_LOG%"

if exist "%GRIM_REPO_ROOT%\.venv\Scripts\python.exe" (
    call :try_python "%GRIM_REPO_ROOT%\.venv\Scripts\python.exe"
    if not errorlevel 1 goto launch_python
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    call :try_python "%VIRTUAL_ENV%\Scripts\python.exe"
    if not errorlevel 1 goto launch_python
)

where py.exe >nul 2>&1
if not errorlevel 1 (
    echo --- py.exe -3 --- >>"%FREDDY_LAUNCH_LOG%"
    py.exe -3 -c "import numpy, scipy; from ibc.ui import QT_AVAILABLE, MPL_AVAILABLE; assert QT_AVAILABLE, 'PySide6 is not available'; assert MPL_AVAILABLE, 'The Matplotlib Qt backend is not available'" >>"%FREDDY_LAUNCH_LOG%" 2>&1
    if not errorlevel 1 goto launch_py
)

where python.exe >nul 2>&1
if not errorlevel 1 (
    call :try_python "python.exe"
    if not errorlevel 1 goto launch_python
)

goto missing_dependencies

:try_python
set "FREDDY_PYTHON=%~1"
echo --- %FREDDY_PYTHON% --- >>"%FREDDY_LAUNCH_LOG%"
"%FREDDY_PYTHON%" -c "import numpy, scipy; from ibc.ui import QT_AVAILABLE, MPL_AVAILABLE; assert QT_AVAILABLE, 'PySide6 is not available'; assert MPL_AVAILABLE, 'The Matplotlib Qt backend is not available'" >>"%FREDDY_LAUNCH_LOG%" 2>&1
exit /b %ERRORLEVEL%

:launch_python
for %%I in ("%FREDDY_PYTHON%") do set "FREDDY_PYTHONW=%%~dpIpythonw.exe"
if exist "%FREDDY_PYTHONW%" (
    start "" "%FREDDY_PYTHONW%" "%~dp0impedance_gui.py"
) else (
    start "" "%FREDDY_PYTHON%" "%~dp0impedance_gui.py"
)
exit /b 0

:launch_py
where pyw.exe >nul 2>&1
if errorlevel 1 (
    start "" py.exe -3 "%~dp0impedance_gui.py"
) else (
    start "" pyw.exe -3 "%~dp0impedance_gui.py"
)
exit /b 0

:missing_dependencies
echo ERROR: No preferred Python interpreter could import the FREDDY GUI.
echo.
if exist "%FREDDY_LAUNCH_LOG%" type "%FREDDY_LAUNCH_LOG%"
echo.
echo From the repository root, create the shared environment and install with:
echo     py.exe -3 -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -e .
echo.
pause
exit /b 1
