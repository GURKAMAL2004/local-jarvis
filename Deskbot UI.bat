@echo off
setlocal enableextensions
title deskbot - Web Interface
color 0B

echo ============================================
echo    deskbot - Web Interface
echo ============================================
echo.
echo Starting the local web interface...
echo A browser tab will open automatically in a moment.
echo Close this window to stop the server.
echo.

where deskbot >nul 2>nul
if %errorlevel%==0 (
    set "DESKBOT=deskbot"
) else (
    set "DESKBOT="D:\LOCAL LLM TO USE SHIT\.venv\Scripts\deskbot.exe""
)

%DESKBOT% ui

pause
