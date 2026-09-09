@echo off
setlocal
title GEDICorrect v1.0.0
pushd "%~dp0" >nul
if errorlevel 1 (
    echo GEDICorrect could not access its installation folder.
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-GEDICorrect.ps1"
set "GEDICORRECT_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%GEDICORRECT_EXIT_CODE%"=="0" (
    echo.
    echo GEDICorrect could not be started.
    pause
)

exit /b %GEDICORRECT_EXIT_CODE%
