#!/bin/bash
# Mac/Linux Shell Script to Convert All Markdown to HTML
# Run: bash convert_all.sh

echo "========================================"
echo "  Markdown to HTML Converter"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null
then
    echo "❌ ERROR: Python is not installed"
    echo "Please install Python from https://python.org"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null
then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "Converting all markdown files..."
echo ""

$PYTHON_CMD convert_all_md.py

echo ""
echo "========================================"
echo "  ✅ Conversion Complete!"
echo "========================================"
echo ""
echo "All .html files have been created."
echo "Open any .html file in your browser."
echo ""
