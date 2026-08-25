@echo off
setlocal
cd /d "%~dp0"

if not exist "GRIM_Revised_2\grim_cut_gui.py" (
    echo ERROR: GRIM_Revised_2\grim_cut_gui.py was not found.
    echo Keep the complete branch folder structure together.
    pause
    exit /b 1
)

set "GRIM_LAUNCH_LOG=%TEMP%\grim-gui-launch.log"
type nul >"%GRIM_LAUNCH_LOG%"

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
    echo --- py.exe -3 --- >>"%GRIM_LAUNCH_LOG%"
    py.exe -3 -c "import sys; sys.path.insert(0, r'%~dp0GRIM_Revised_2'); import grim_cut_gui" >>"%GRIM_LAUNCH_LOG%" 2>&1
    if not errorlevel 1 goto launch_py
)

where python.exe >nul 2>&1
if not errorlevel 1 (
    call :try_python "python.exe"
    if not errorlevel 1 goto launch_python
)

goto missing_dependencies

:try_python
set "GRIM_PYTHON=%~1"
echo --- %GRIM_PYTHON% --- >>"%GRIM_LAUNCH_LOG%"
"%GRIM_PYTHON%" -c "import sys; sys.path.insert(0, r'%~dp0GRIM_Revised_2'); import grim_cut_gui" >>"%GRIM_LAUNCH_LOG%" 2>&1
exit /b %ERRORLEVEL%

:launch_python
for %%I in ("%GRIM_PYTHON%") do set "GRIM_PYTHONW=%%~dpIpythonw.exe"
if exist "%GRIM_PYTHONW%" (
    start "" "%GRIM_PYTHONW%" "%~dp0GRIM_Revised_2\grim_cut_gui.py"
) else (
    start "" "%GRIM_PYTHON%" "%~dp0GRIM_Revised_2\grim_cut_gui.py"
)
exit /b 0

:launch_py
where pyw.exe >nul 2>&1
if errorlevel 1 (
    start "" py.exe -3 "%~dp0GRIM_Revised_2\grim_cut_gui.py"
) else (
    start "" pyw.exe -3 "%~dp0GRIM_Revised_2\grim_cut_gui.py"
)
exit /b 0

:missing_dependencies
echo ERROR: No preferred Python interpreter could import GRIM.
echo.
if exist "%GRIM_LAUNCH_LOG%" type "%GRIM_LAUNCH_LOG%"
echo.
echo Create the shared environment at %~dp0.venv, then install with:
echo     .venv\Scripts\python.exe -m pip install -e .
echo.
echo The launcher checks the repository .venv first, then VIRTUAL_ENV,
echo then the system Python launchers.
pause
exit /b 1
