FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Ferramentas base de empacotamento
RUN python -m pip install --upgrade pip setuptools wheel

# Instalar PyTorch com CUDA primeiro para evitar fallback CPU
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar dependências Python restantes
RUN python -m pip install -r requirements.txt

# Copiar código da aplicação
COPY . .

# Expor porta do Streamlit
EXPOSE 8501

# Comando para iniciar o Streamlit
CMD ["python", "-m", "streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
