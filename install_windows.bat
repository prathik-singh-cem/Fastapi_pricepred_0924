@echo off
echo Installing T-Mobile Installation Cost Prediction API Dependencies...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install core FastAPI dependencies first
echo Installing FastAPI and core dependencies...
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6

REM Install authentication dependencies
echo Installing authentication dependencies...
pip install PyJWT==2.8.0
pip install passlib[bcrypt]==1.7.4
pip install cryptography==41.0.7

REM Install configuration management
echo Installing configuration dependencies...
pip install pydantic==2.5.0
pip install pydantic-settings==2.1.0
pip install python-dotenv==1.0.0

REM Install requests for testing
echo Installing HTTP client...
pip install requests>=2.28.0

REM Install ML dependencies using pre-compiled wheels
echo Installing ML dependencies (this may take a moment)...
pip install --only-binary=all pandas>=2.0.0,^<3.0.0
pip install --only-binary=all numpy>=1.21.0,^<2.0.0
pip install --only-binary=all scikit-learn>=1.3.0,^<2.0.0
pip install --only-binary=all xgboost>=1.7.0,^<3.0.0

echo.
echo Installation completed successfully!
echo.
echo To start the API server, run:
echo uvicorn main:app --reload
echo.
echo To test the API, run:
echo python test_api.py
echo.
pause
