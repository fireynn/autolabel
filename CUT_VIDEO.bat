@echo off
rem Cut video into frames. Drag a video (or a folder with videos) onto this file,
rem or run:  CUT_VIDEO.bat video.mp4 --fps 2
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" "%~dp0video_to_frames.py" %*
pause
