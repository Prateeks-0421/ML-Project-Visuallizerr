@echo off
REM Advanced ML Workbench - Windows Setup Script

echo.
echo 🚀 Advanced ML Workbench Pro - Setup Script
echo ===========================================
echo.

REM Check Python version
echo ✓ Checking Python version...
python --version
echo.

REM Create virtual environment
echo ✓ Creating virtual environment...
python -m venv venv
echo   Virtual environment created
echo.

REM Activate virtual environment
echo ✓ Activating virtual environment...
call venv\Scripts\activate.bat
echo   Virtual environment activated
echo.

REM Upgrade pip
echo ✓ Upgrading pip...
python -m pip install --upgrade pip --quiet
echo   pip upgraded
echo.

REM Install dependencies
echo ✓ Installing dependencies...
pip install -r requirements.txt --quiet
echo   Dependencies installed:
echo   - streamlit
echo   - pandas
echo   - numpy
echo   - scikit-learn
echo   - matplotlib
echo   - seaborn
echo   - joblib
echo.

REM Create necessary directories
echo ✓ Creating directories...
if not exist "models" mkdir models
if not exist "data" mkdir data
if not exist "reports" mkdir reports
echo   Directories created:
echo   - models/
echo   - data/
echo   - reports/
echo.

REM Verify installation
echo ✓ Verifying installation...
python -c "import streamlit; import pandas; import sklearn; print('  All packages verified!')"
echo.

REM Print next steps
echo ✅ Setup complete!
echo.
echo 📖 Next steps:
echo 1. Activate virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Run the application:
echo    streamlit run ml_workbench_advanced.py
echo.
echo 3. Open browser to:
echo    http://localhost:8501
echo.
echo 📚 Documentation:
echo    - README.md - Main documentation
echo    - ADVANCED_GUIDE.md - Advanced features
echo.
echo 🎉 Happy ML modeling!
echo.
pause
