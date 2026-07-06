@echo off
setlocal enableextensions
title deskbot - Deep Research
color 0B

echo ============================================
echo    deskbot - Deep Research
echo ============================================
echo.
echo This will search the web, read several real sources,
echo and write you a summarized report you can read afterward.
echo.

where deskbot >nul 2>nul
if %errorlevel%==0 (
    set "DESKBOT=deskbot"
) else (
    set "DESKBOT="D:\LOCAL LLM TO USE SHIT\.venv\Scripts\deskbot.exe""
)

set /p TOPIC="What do you want to research? "
if "%TOPIC%"=="" (
    echo You didn't type anything - closing.
    pause
    exit /b 1
)

echo.
echo Next you'll be asked to pick a research method (Quick/Standard/Deep/Custom)
echo and, if you want, specific models - just press Enter on any question to
echo use the sensible default.
echo.
echo Once that's done: researching can take a few minutes, and a browser
echo window will open by itself - that's normal, let it work.
echo.

%DESKBOT% research "%TOPIC%"

echo.
echo ============================================
echo Done. The report has also been saved to your
echo Desktop, in a folder called "deskbot-research".
echo ============================================
echo.
pause
