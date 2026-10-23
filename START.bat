@echo off
rem ================================================================
rem  Autolabel Studio - install (first run) and start
rem  Double-click. Everything downloads automatically, no proxy.
rem  Options:  START.bat --preset best    START.bat --cpu
rem ================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "HTTP_PROXY=" & set "HTTPS_PROXY=" & set "ALL_PROXY=" & set "http_proxy=" & set "https_proxy=" & set "all_proxy=" & set "PIP_PROXY="
set "NO_PROXY=*" & set "AUTOLABEL_NO_PROXY=1" & set "PYTHONIOENCODING=utf-8"

if exist ".venv\.installed_ok" if exist ".venv\Scripts\python.exe" if "%~1"=="" goto run

rem ---------- find Python 3.9 - 3.12 ----------
set "PY="
for %%V in (3.11 3.12 3.10 3.9) do (
  if not defined PY ( py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V" )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if not defined PY (
  echo Python 3.11 not found - downloading from python.org ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; [System.Net.WebRequest]::DefaultWebProxy=$null; Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile \"$env:TEMP\python-3.11.9-amd64.exe\""
  if not exist "%TEMP%\python-3.11.9-amd64.exe" (
    echo [ERROR] Could not download Python. Check the internet connection.
    pause
    exit /b 1
  )
  echo Installing Python 3.11 ...
  "%TEMP%\python-3.11.9-amd64.exe" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0
  if not exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    echo [ERROR] Python installation failed.
    pause
    exit /b 1
  )
  set PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

%PY% "%~dp0install.py" %*
if errorlevel 1 (
  echo.
  echo [ERROR] Installation failed - see messages above. Run START.bat again to continue.
  pause
  exit /b 1
)

:run
echo Starting Autolabel Studio ...
".venv\Scripts\python.exe" "%~dp0autolabel_app.py"
if errorlevel 1 pause
