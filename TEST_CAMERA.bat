@echo off
rem Test the trained model on a webcam. Put yolov4-tiny-custom_best.weights,
rem yolov4-tiny-custom.cfg and obj.names into the "model" folder next to this file.
rem Options: TEST_CAMERA.bat --cam 1     TEST_CAMERA.bat --source video.mp4
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
  echo Run START.bat first - it installs everything.
  pause
  exit /b 1
)
if not exist "model" mkdir model
".venv\Scripts\python.exe" "%~dp0webcam_test.py" %*
if errorlevel 1 pause
