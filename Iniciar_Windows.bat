@echo off
title Pronto-Socorro - Triagem IA
cd /d "%~dp0"

if not exist "venv" (
    echo Preparando ambiente na primeira execucao...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

start "" venv\Scripts\pythonw.exe agente_urgencia.py