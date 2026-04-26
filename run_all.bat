@echo off
cd /d "%~dp0"
set PYTHON=C:\Users\ouyan\AppData\Local\Programs\Python\Python312\python.exe

echo ==========================================
echo   Japan Inflation Assemblage - Run All
echo ==========================================

echo.
echo --- Component (level 1) ---
%PYTHON% regression/regression_component.py --level 1
if errorlevel 1 goto error

echo.
echo --- Component (level 2) ---
%PYTHON% regression/regression_component.py --level 2
if errorlevel 1 goto error

echo.
echo --- Component (level 3)  [slow: ~700 components] ---
%PYTHON% regression/regression_component.py --level 3
if errorlevel 1 goto error

echo.
echo --- Ranks (level 1) ---
%PYTHON% regression/regression_rank.py --level 1
if errorlevel 1 goto error

echo.
echo --- Ranks (level 2) ---
%PYTHON% regression/regression_rank.py --level 2
if errorlevel 1 goto error

echo.
echo --- Ranks (level 3)  [slow] ---
%PYTHON% regression/regression_rank.py --level 3
if errorlevel 1 goto error

echo.
echo All done. Plots saved to plots\
pause
exit /b 0

:error
echo.
echo ERROR: script failed. See output above.
pause
exit /b 1
