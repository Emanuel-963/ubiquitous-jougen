#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# IonFlow Pipeline — Instalador Linux / macOS
# ──────────────────────────────────────────────────────────────
# Uso:
#   chmod +x install_linux.sh
#   ./install_linux.sh
#
# O que este script faz:
#   1. Verifica se Python 3.11+ está instalado
#   2. Cria um ambiente virtual (venv)
#   3. Instala todas as dependências
#   4. Cria as pastas de dados e saída
#   5. Cria um atalho de desktop (opcional)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Cores ────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✔]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✖]${NC} $*"; exit 1; }

# ── Diretório do script ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   IonFlow Pipeline — Instalador Linux/macOS  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Encontrar Python 3.11+ ───────────────────────────────
PYTHON=""
for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major="${version%%.*}"
        minor="${version#*.}"
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    warn "Python 3.11+ não encontrado."
    echo ""
    echo "  Instale com um dos comandos abaixo:"
    echo ""
    echo "  Ubuntu/Debian:"
    echo "    sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-tk"
    echo ""
    echo "  Fedora:"
    echo "    sudo dnf install -y python3.11 python3-tkinter"
    echo ""
    echo "  Arch:"
    echo "    sudo pacman -S python tk"
    echo ""
    echo "  macOS (Homebrew):"
    echo "    brew install python@3.12 python-tk@3.12"
    echo ""
    error "Instale Python 3.11+ e rode este script novamente."
fi

info "Python encontrado: $PYTHON ($($PYTHON --version 2>&1))"

# ── 2. Verificar tkinter ────────────────────────────────────
if ! "$PYTHON" -c "import tkinter" &>/dev/null; then
    warn "tkinter não está instalado (necessário para a GUI)."
    echo ""
    echo "  Ubuntu/Debian:  sudo apt install -y python3-tk"
    echo "  Fedora:         sudo dnf install -y python3-tkinter"
    echo "  macOS:          brew install python-tk@3.12"
    echo ""
    error "Instale tkinter e rode novamente."
fi
info "tkinter disponível."

# ── 3. Criar ambiente virtual ────────────────────────────────
if [ -d "venv" ]; then
    # Verificar se o venv funciona (pode ter sido copiado de outro PC/Windows)
    if [ -x "./venv/bin/python" ]; then
        if ./venv/bin/python -c "print('ok')" &>/dev/null; then
            info "Ambiente virtual já existe e está funcional."
        else
            warn "Removendo venv antigo (incompatível/copiado de outro computador)..."
            rm -rf venv
            info "Criando ambiente virtual novo..."
            "$PYTHON" -m venv venv
        fi
    else
        warn "Removendo venv inválido (provavelmente de Windows)..."
        rm -rf venv
        info "Criando ambiente virtual novo..."
        "$PYTHON" -m venv venv
    fi
else
    info "Criando ambiente virtual..."
    "$PYTHON" -m venv venv
fi

VENV_PYTHON="./venv/bin/python"
info "Atualizando pip..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

# ── 4. Instalar dependências ────────────────────────────────
info "Instalando dependências..."
"$VENV_PYTHON" -m pip install . --quiet

if [ -f "requirements-dev.txt" ]; then
    echo -n "  Deseja instalar dependências de desenvolvimento (testes, lint)? [s/N] "
    read -r dev_choice
    if [[ "$dev_choice" =~ ^[sS]$ ]]; then
        "$VENV_PYTHON" -m pip install -r requirements-dev.txt --quiet
        info "Dependências de desenvolvimento instaladas."
    fi
fi

# ── 5. Criar pastas ─────────────────────────────────────────
for dir in data/raw data/processed outputs/figures outputs/tables outputs/excel outputs/circuit_reports; do
    mkdir -p "$dir"
done
info "Pastas de dados e saída criadas."

# ── 6. Atalho de desktop (opcional) ─────────────────────────
DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
if [ -d "$DESKTOP_DIR" ]; then
    echo -n "  Criar atalho na Área de Trabalho? [s/N] "
    read -r shortcut_choice
    if [[ "$shortcut_choice" =~ ^[sS]$ ]]; then
        DESKTOP_FILE="$DESKTOP_DIR/IonFlow_Pipeline.desktop"
        cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=IonFlow Pipeline
Comment=EIS Analytics Toolkit
Exec=bash -c 'cd "$SCRIPT_DIR" && ./venv/bin/python gui_app.py'
Icon=$SCRIPT_DIR/data/ionflow.ico
Terminal=false
Categories=Science;Education;
EOF
        chmod +x "$DESKTOP_FILE"
        info "Atalho criado em $DESKTOP_FILE"
    fi
fi

# ── 7. Resumo ───────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║       Instalação concluída com sucesso!       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Para abrir o programa:"
echo ""
echo "    cd $SCRIPT_DIR"
echo "    source venv/bin/activate"
echo "    python gui_app.py"
echo ""
echo "  Ou com um único comando:"
echo ""
echo "    $SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/gui_app.py"
echo ""
