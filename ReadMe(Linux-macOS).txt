╔══════════════════════════════════════════════════════════════╗
║              IonFlow Pipeline  v0.1.0                       ║
║    Análise de Espectroscopia de Impedância Eletroquímica    ║
║                    Linux / macOS                            ║
╚══════════════════════════════════════════════════════════════╝

Obrigado por usar o IonFlow Pipeline!
Este guia explica como instalar e usar o programa no Linux ou macOS.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. REQUISITOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Python 3.11 ou superior
  • tkinter (normalmente já vem com Python)
  • git (para baixar o código)
  • Conexão com a internet (apenas na instalação)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2. INSTALAR PYTHON (se ainda não tiver)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Ubuntu / Debian:
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3-tk git

  Fedora:
    sudo dnf install -y python3.11 python3-tkinter git

  Arch Linux:
    sudo pacman -S python tk git

  macOS (Homebrew):
    brew install python@3.12 python-tk@3.12 git

  Para verificar se já tem:
    python3 --version


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3. INSTALAÇÃO (copie e cole no terminal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Passo 1 — Baixar o código do GitHub
  git clone https://github.com/Emanuel-963/ubiquitous-jougen.git
  cd ubiquitous-jougen

  # Passo 2 — Rodar o instalador automático
  chmod +x install_linux.sh
  ./install_linux.sh

  O script vai:
    ✔ Verificar se Python 3.11+ está instalado
    ✔ Criar um ambiente virtual (venv)
    ✔ Instalar todas as dependências
    ✔ Criar as pastas de dados e saída
    ✔ Perguntar se quer criar atalho na Área de Trabalho


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  4. COMO ABRIR O PROGRAMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Opção A — Dois comandos:

    cd ubiquitous-jougen
    source venv/bin/activate
    python gui_app.py

  Opção B — Comando único (sem ativar venv):

    ./ubiquitous-jougen/venv/bin/python ./ubiquitous-jougen/gui_app.py

  Opção C — Se criou o atalho de desktop:

    Dê duplo-clique em "IonFlow Pipeline" na Área de Trabalho.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  5. COMO USAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A) IMPORTAR DADOS

     Na barra lateral esquerda:

     • "Importar EIS para raw"
       → Selecione seus arquivos .txt de impedância (EIS)

     • "Importar Ciclagem para processed"
       → Selecione seus arquivos .txt de ciclagem galvanostática

  B) RODAR ANÁLISES

     • "Rodar Pipeline EIS"      → analisa dados de impedância
     • "Rodar Pipeline Ciclagem" → analisa dados de ciclagem
     • "Rodar Ambos"             → roda EIS + Ciclagem
     • "Rodar Pipeline DRT"      → Distribution of Relaxation Times

  C) VER RESULTADOS

     • Aba "Gráficos"  → imagens geradas
     • Aba "Tabelas"   → dados numéricos
     • Aba "Logs"      → mensagens do processamento
     • "Gráficos Interativos" → 17 abas com zoom, pan e hover

  D) SALVAR

     • Botões "Salvar imagem" e "Salvar tudo" em cada aba
     • Resultados ficam em: outputs/


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  6. LINHA DE COMANDO (sem GUI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Coloque os arquivos .txt em data/raw/ (EIS) ou
  data/processed/ (ciclagem), depois rode:

    python main.py              # Pipeline EIS
    python main_cycling.py      # Pipeline Ciclagem
    python main_drt.py          # Pipeline DRT


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  7. PROBLEMAS COMUNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  P: "ModuleNotFoundError: No module named 'tkinter'"
  R: Instale tkinter:
     Ubuntu:  sudo apt install python3-tk
     Fedora:  sudo dnf install python3-tkinter
     macOS:   brew install python-tk@3.12

  P: "Python 3.11+ não encontrado"
  R: Instale Python 3.11 ou superior (veja seção 2 acima).

  P: A GUI abre mas a tela fica em branco.
  R: Aguarde alguns segundos. Se persistir, verifique os Logs.

  P: Erro de permissão ao rodar install_linux.sh
  R: Execute: chmod +x install_linux.sh

  P: O venv não funciona (erros ao ativar ou importar módulos).
  R: Se a pasta veio copiada de outro computador (Windows, OneDrive,
     USB), o venv antigo tem caminhos incompatíveis. O instalador
     agora detecta e recria automaticamente. Se preferir fazer
     manualmente:
       rm -rf venv
       ./install_linux.sh

  P: Onde ficam os resultados?
  R:  outputs/figures/   → gráficos em .png
      outputs/tables/    → tabelas em .csv
      outputs/excel/     → planilhas consolidadas


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  8. ATUALIZAR PARA NOVA VERSÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd ubiquitous-jougen
  git pull
  source venv/bin/activate
  pip install -r requirements.txt


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  9. CONTATO E CÓDIGO-FONTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GitHub: https://github.com/Emanuel-963/ubiquitous-jougen
  Licença: MIT (uso livre)

  Desenvolvido por Emanuel — Iniciação Científica
  Análise de materiais eletroquímicos via EIS, DRT e ciclagem.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━