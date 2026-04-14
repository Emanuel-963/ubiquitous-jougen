#!/usr/bin/env bash
set -euo pipefail

# Instalador automatizado para o pipeline de Ciclagem e/ou EIS (Linux)

CICLAGEM=false
EIS=false

for arg in "$@"; do
  case "$arg" in
    --ciclagem) CICLAGEM=true ;;
    --eis) EIS=true ;;
    --ambos) CICLAGEM=true; EIS=true ;;
    *) ;; 
  esac
done

if [ "$CICLAGEM" = false ] && [ "$EIS" = false ]; then
  echo "--- Instalador do Pipeline (Linux) ---"
  echo "Escolha o pipeline para instalar:"
  echo "1 - Ciclagem"
  echo "2 - EIS"
  echo "3 - Ambos"
  read -r -p "Digite o número da opção desejada: " choice
  case "$choice" in
    1) CICLAGEM=true ;;
    2) EIS=true ;;
    3) CICLAGEM=true; EIS=true ;;
    *) echo "Opção inválida."; exit 1 ;;
  esac
fi

install_python311() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo "Python 3.11 já instalado."
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    echo "Instalando Python 3.11 via apt-get..."
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils
    return 0
  fi

  echo "Não foi possível instalar Python 3.11 automaticamente."
  echo "Instale o Python 3.11 manualmente e rode novamente este script."
  exit 1
}

create_venv() {
  if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python3.11 -m venv venv
  else
    echo "Ambiente virtual já existe."
  fi
}

install_requirements() {
  local venv_python="./venv/bin/python"
  echo "Instalando dependências via pyproject.toml..."
  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install .
  if [ -f "requirements-dev.txt" ]; then
    "$venv_python" -m pip install -r requirements-dev.txt
  fi
}

setup_folders() {
  echo "Criando pastas do pipeline..."
  local folders=(
    "data/raw"
    "data/processed"
    "outputs/figures"
    "outputs/tables"
    "outputs/excel"
  )
  for f in "${folders[@]}"; do
    mkdir -p "$f"
  done
}

setup_envvars() {
  local venv_path
  venv_path="$(pwd)/venv"
  if ! grep -q "VIRTUAL_ENV=.*${venv_path}" "$HOME/.bashrc" 2>/dev/null; then
    {
      echo ""
      echo "# eis_analytics venv"
      echo "export VIRTUAL_ENV=\"${venv_path}\""
      echo "export PATH=\"${venv_path}/bin:\$PATH\""
    } >> "$HOME/.bashrc"
  fi
}

install_python311
create_venv
install_requirements

if [ "$CICLAGEM" = true ] || [ "$EIS" = true ]; then
  setup_folders
fi

setup_envvars

echo ""
echo "Instalação concluída!"
echo "Ative o ambiente com: source ./venv/bin/activate"
echo "Depois rode: python main_cycling.py (ou o script do pipeline EIS)"
