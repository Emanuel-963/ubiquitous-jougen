#!/bin/bash

echo "======================================"
echo " Setting up EIS Analytics Environment "
echo "======================================"

# Verifica Python
python --version || {
  echo "Python não encontrado. Instale Python >= 3.9."
  exit 1
}

# Cria ambiente virtual
if [ ! -d "venv" ]; then
  echo "Criando ambiente virtual..."
  python -m venv venv
fi

# Ativa ambiente virtual
source venv/bin/activate

# Atualiza pip
pip install --upgrade pip

# Instala dependências
pip install \
  numpy \
  pandas \
  matplotlib \
  scikit-learn \
  scipy

echo "======================================"
echo " Ambiente configurado com sucesso ✅"
echo " Para ativar: source venv/bin/activate"
echo "======================================"
