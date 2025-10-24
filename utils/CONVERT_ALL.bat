@echo off
REM Windows Batch File to Convert All Markdown to HTML
REM Double-click this file to convert all .md files

echo ========================================
echo   Markdown to HTML Converter
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Converting all markdown files...
echo.

python convert_all_md.py

echo.
echo ========================================
echo   Conversion Complete!
echo ========================================
echo.
echo All .html files have been created.
echo Double-click any .html file to open in browser.
echo.

pause
