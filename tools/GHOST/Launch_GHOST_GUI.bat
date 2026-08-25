@echo off
setlocal
cd /d "%~dp0"

if not exist "Backend\ghost_gui.py" (
    echo ERROR: Backend\ghost_gui.py was not found.
    pause
    exit /b 1
)

for %%I in ("%~dp0..\..") do set "GRIM_REPO_ROOT=%%~fI"
set "GHOST_LAUNCH_LOG=%TEMP%\ghost-gui-launch.log"
type nul >"%GHOST_LAUNCH_LOG%"

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
    echo --- py.exe -3 --- >>"%GHOST_LAUNCH_LOG%"
    py.exe -3 "Backend\ghost_gui.py" --check >>"%GHOST_LAUNCH_LOG%" 2>&1
    if not errorlevel 1 goto launch_py
)

where python.exe >nul 2>&1
if not errorlevel 1 (
    call :try_python "python.exe"
    if not errorlevel 1 goto launch_python
)

goto missing_dependencies

:try_python
set "GHOST_PYTHON=%~1"
echo --- %GHOST_PYTHON% --- >>"%GHOST_LAUNCH_LOG%"
"%GHOST_PYTHON%" "Backend\ghost_gui.py" --check >>"%GHOST_LAUNCH_LOG%" 2>&1
exit /b %ERRORLEVEL%

:launch_python
for %%I in ("%GHOST_PYTHON%") do set "GHOST_PYTHONW=%%~dpIpythonw.exe"
if exist "%GHOST_PYTHONW%" (
    start "" "%GHOST_PYTHONW%" "%~dp0Backend\ghost_gui.py"
) else (
    start "" "%GHOST_PYTHON%" "%~dp0Backend\ghost_gui.py"
)
exit /b 0

:launch_py
where pyw.exe >nul 2>&1
if errorlevel 1 (
    start "" py.exe -3 "%~dp0Backend\ghost_gui.py"
) else (
    start "" pyw.exe -3 "%~dp0Backend\ghost_gui.py"
)
exit /b 0

:missing_dependencies
echo ERROR: No preferred Python interpreter could import the GHOST GUI.
echo.
if exist "%GHOST_LAUNCH_LOG%" type "%GHOST_LAUNCH_LOG%"
echo.
echo From the repository root, create the shared environment and install with:
echo     py.exe -3 -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -e .
echo.
pause
exit /b 1
