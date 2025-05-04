#!/bin/bash
echo "BhedChaal - CCTV Analysis Tool"
echo "============================="
echo ""
echo "1. Basic Mode"
echo "2. Advanced Mode"
echo "3. Interactive Mode (mouse point selection)"
echo "4. Simple Interactive Mode (more compatible)"
echo ""

read -p "Choose a mode (1, 2, 3, or 4): " choice

if [ "$choice" = "4" ]; then
    echo "Starting Simple Interactive Mode..."
    python run_app.py --simple
elif [ "$choice" = "3" ]; then
    echo "Starting Interactive Mode..."
    python run_app.py --interactive
elif [ "$choice" = "2" ]; then
    echo "Starting Advanced Mode..."
    python run_app.py --advanced
else
    echo "Starting Basic Mode..."
    python run_app.py 