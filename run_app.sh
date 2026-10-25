#!/usr/bin/env bash
# ==============================================================================
# Quick Launcher: GEO Healthcare AI Interactive Prototype (Linux / macOS)
# SIT Tumakuru - V Semester Mini-Project
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Starting Generative Engine Optimization (GEO) Healthcare AI Prototype..."
streamlit run app.py --server.fileWatcherType=none
