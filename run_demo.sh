#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Check if venv exists, if not, create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the demo
echo "Running ScamDefy Demo..."
python3 demo.py
