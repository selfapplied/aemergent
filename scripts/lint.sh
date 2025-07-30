#!/bin/bash
# Lint script for aemergent

echo "🔍 Running flake8 linting..."
source .venv/bin/activate
python3 -m flake8 src/ demos/ --max-line-length=100 --count --statistics

if [ $? -eq 0 ]; then
    echo "✅ Linting passed!"
    exit 0
else
    echo "❌ Linting failed!"
    exit 1
fi 