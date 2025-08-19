@echo off
REM === Ajustar a política de execução para o PowerShell ===
powershell -Command "Set-ExecutionPolicy RemoteSigned -Scope Process -Force"

REM === Ativar o ambiente virtual ===
call .\.filesvenv\Scripts\activate.bat

REM === Entrar na pasta src ===
cd src

REM === Rodar o Streamlit ===
streamlit run main.py
