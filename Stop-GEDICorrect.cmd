@echo off
setlocal
pushd "%~dp0" >nul
if errorlevel 1 (
    echo GEDICorrect could not access its installation folder.
    pause
    exit /b 1
)

docker compose stop
set "GEDICORRECT_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%GEDICORRECT_EXIT_CODE%"=="0" (
    echo.
    echo GEDICorrect could not be stopped. Make sure Docker Desktop is running.
    pause
    exit /b %GEDICORRECT_EXIT_CODE%
)

echo.
echo GEDICorrect has stopped. Your input and output files were not removed.
timeout /t 3 >nul
exit /b 0
