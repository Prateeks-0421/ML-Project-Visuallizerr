#!/bin/bash

# Advanced ML Workbench - Setup Script
# This script automates the setup process

echo "🚀 Advanced ML Workbench Pro - Setup Script"
echo "==========================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"
echo ""

# Create virtual environment
echo "✓ Creating virtual environment..."
python3 -m venv venv
echo "  Virtual environment created"
echo ""

# Activate virtual environment
echo "✓ Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/macOS
    source venv/bin/activate
fi
echo "  Virtual environment activated"
echo ""

# Upgrade pip
echo "✓ Upgrading pip..."
python3 -m pip install --upgrade pip --quiet
echo "  pip upgraded"
echo ""

# Install dependencies
echo "✓ Installing dependencies..."
pip install -r requirements.txt --quiet
echo "  Dependencies installed:"
echo "  - streamlit"
echo "  - pandas"
echo "  - numpy"
echo "  - scikit-learn"
echo "  - matplotlib"
echo "  - seaborn"
echo "  - joblib"
echo ""

# Create necessary directories
echo "✓ Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p reports
echo "  Directories created:"
echo "  - models/"
echo "  - data/"
echo "  - reports/"
echo ""

# Verify installation
echo "✓ Verifying installation..."
python3 -c "import streamlit; import pandas; import sklearn; print('  All packages verified!')"
echo ""

# Print next steps
echo "✅ Setup complete!"
echo ""
echo "📖 Next steps:"
echo "1. Activate virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "   venv\\Scripts\\activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Run the application:"
echo "   streamlit run ml_workbench_advanced.py"
echo ""
echo "3. Open browser to:"
echo "   http://localhost:8501"
echo ""
echo "📚 Documentation:"
echo "   - README.md - Main documentation"
echo "   - ADVANCED_GUIDE.md - Advanced features"
echo ""
echo "🎉 Happy ML modeling!"
