@echo off
REM === Ajustar a política de execução para o PowerShell ===
powershell -Command "Set-ExecutionPolicy RemoteSigned -Scope Process -Force"

REM === Ativar o ambiente virtual ===
call .\.filesvenv\Scripts\activate.bat

REM === Entrar na pasta src ===
cd src

REM === Rodar o Streamlit ===
python relay_rtsp_server.py "rtsp://admin:admin123@172.16.5.250:554/Streaming/Channels/101" --host 0.0.0.0 --port 5555 --quality 85 --send-fps 15
