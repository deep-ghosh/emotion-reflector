@echo off
echo 🛑 Stopping all emotion detection processes...
taskkill /F /IM python.exe 2>nul
if errorlevel 1 (
    echo ℹ️ No Python processes found running
) else (
    echo ✅ All emotion detection processes stopped
)
echo.
echo Press any key to continue...
pause >nul
