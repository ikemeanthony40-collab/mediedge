@echo off
title MediEdge Launcher
color 0A

echo.
echo  Starting MediEdge...
echo.

:: Check Ollama
echo [1/3] Starting Ollama...
start /min "" "C:\Users\pc\AppData\Local\Programs\Ollama\ollama.exe" serve
timeout /t 3 /nobreak > nul

:: Start backend
echo [2/3] Starting backend...
start /min cmd /k "cd /d C:\Users\pc\mediEdge\mediEdge\backend && python main.py"
timeout /t 4 /nobreak > nul

:: Start frontend
echo [3/3] Starting frontend...
start /min cmd /k "cd /d C:\Users\pc\mediEdge\mediEdge\pwa\public && python -m http.server 5500"
timeout /t 2 /nobreak > nul

:: Open browser
echo Opening MediEdge...
start chrome "http://localhost:5500"

echo.
echo  MediEdge is running!
echo  App: http://localhost:5500
echo  API: http://localhost:8000
echo.
echo  Press any key to stop all services...
pause > nul

taskkill /f /im python.exe > nul 2>&1
echo MediEdge stopped.