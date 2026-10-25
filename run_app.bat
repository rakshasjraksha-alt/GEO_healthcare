@echo off
REM ==============================================================================
REM Quick Launcher: GEO Healthcare AI Interactive Prototype (Windows)
REM SIT Tumakuru - V Semester Mini-Project
REM ==============================================================================

echo ==============================================================================
echo Starting Generative Engine Optimization (GEO) Healthcare AI Prototype...
echo Department of CSE (AI ^& ML), Siddaganga Institute of Technology, Tumakuru
echo ==============================================================================

cd /d "%~dp0"

IF EXIST venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

streamlit run app.py --server.fileWatcherType=none
pause
