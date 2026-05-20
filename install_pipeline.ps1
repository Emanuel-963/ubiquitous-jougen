#!/usr/bin/env powershell
<#+
.SYNOPSIS
    Instalador e atualizador automatizado para o IonFlow Pipeline.
.DESCRIPTION
    - Cria ambiente virtual isolado com Python 3.11
    - Instala todas as dependências via pyproject.toml
    - Cria as pastas necessárias para análise
    - Com -Update: faz git pull + reinstala dependências (mantém dados do usuário)
.EXAMPLE
    # Primeira instalação
    .\install_pipeline.ps1

    # Atualizar para a última versão
    .\install_pipeline.ps1 -Update

    # Instalar só o pipeline EIS
    .\install_pipeline.ps1 -EIS
#>

param(
    [switch]$Ciclagem,
    [switch]$EIS,
    [switch]$Update
)

# ── Verificar se está na pasta correta ───────────────────────────────
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "ERRO: Execute este script na pasta raiz do IonFlow (onde fica pyproject.toml)." -ForegroundColor Red
    exit 1
}

# Função para baixar e instalar Python 3.11
function Install-Python311 {
    $pythonUrl = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
    $installer = "$env:TEMP\python-3.11.7-amd64.exe"
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) {
        $ver = & python --version 2>&1
        if ($ver -match "3\.1[1-9]") {
            Write-Host "Python $ver já instalado. OK." -ForegroundColor Green
            return
        }
    }
    if (-not (Get-Command python3.11 -ErrorAction SilentlyContinue)) {
        Write-Host "Baixando Python 3.11..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $pythonUrl -OutFile $installer
        Write-Host "Instalando Python 3.11..."
        Start-Process -Wait -FilePath $installer -ArgumentList '/quiet InstallAllUsers=0 PrependPath=1 Include_test=0'
        Remove-Item $installer
        Write-Host "Python 3.11 instalado." -ForegroundColor Green
    } else {
        Write-Host "Python 3.11 já instalado." -ForegroundColor Green
    }
}

# Função para criar ambiente virtual
function Create-Venv {
    $venvPath = "venv"
    if (Test-Path $venvPath) {
        $venvPy = Join-Path $venvPath "Scripts\python.exe"
        if (Test-Path $venvPy) {
            $result = & $venvPy -c "print('ok')" 2>&1
            if ($result -ne "ok") {
                Write-Host "Removendo venv incompatível..." -ForegroundColor Yellow
                Remove-Item -Recurse -Force $venvPath
            } else {
                Write-Host "Ambiente virtual já existe e está funcional." -ForegroundColor Green
                return
            }
        } else {
            Write-Host "Removendo venv incompleto..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $venvPath
        }
    }
    Write-Host "Criando ambiente virtual..." -ForegroundColor Cyan
    python -m venv $venvPath
    if (-not $?) { python3.11 -m venv $venvPath }
}

# Função para instalar dependências
function Install-Requirements {
    $venvPython = ".\venv\Scripts\python.exe"
    Write-Host "Instalando dependências..." -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip --quiet
    & $venvPython -m pip install -e . --quiet
    Write-Host "Dependências instaladas." -ForegroundColor Green
}

# Função para criar pastas dos pipelines
function Setup-Pipelines {
    Write-Host "Configurando pastas..." -ForegroundColor Cyan
    $folders = @(
        "data/raw",
        "data/processed",
        "outputs/figures",
        "outputs/tables",
        "outputs/excel"
    )
    foreach ($f in $folders) {
        if (-not (Test-Path $f)) { New-Item -ItemType Directory -Path $f | Out-Null }
    }
    Write-Host "Pastas criadas." -ForegroundColor Green
}

# ── Modo de ATUALIZAÇÃO (-Update) ────────────────────────────────────
if ($Update) {
    Write-Host ""
    Write-Host "=== IonFlow Pipeline — ATUALIZAÇÃO ===" -ForegroundColor Cyan
    Write-Host ""

    # Verificar se é repositório git
    if (-not (Test-Path ".git")) {
        Write-Host "ERRO: Esta pasta não é um repositório git." -ForegroundColor Red
        Write-Host "Para instalar pela primeira vez, execute: .\install_pipeline.ps1" -ForegroundColor Yellow
        exit 1
    }

    # Mostrar versão atual
    if (Test-Path ".\venv\Scripts\python.exe") {
        $currentVer = & .\venv\Scripts\python.exe -c "import src; print(src.__version__)" 2>&1
        Write-Host "Versão atual: $currentVer" -ForegroundColor White
    }

    Write-Host "Buscando atualizações no GitHub..." -ForegroundColor Cyan
    git fetch origin main 2>&1 | Out-Null

    $behind = git rev-list HEAD..origin/main --count 2>&1
    if ($behind -eq "0") {
        Write-Host "Já está na versão mais recente!" -ForegroundColor Green
        exit 0
    }

    Write-Host "$behind commit(s) novo(s) disponíveis. Atualizando..." -ForegroundColor Yellow
    git pull origin main

    if (-not $?) {
        Write-Host "ERRO: git pull falhou. Verifique se há conflitos." -ForegroundColor Red
        exit 1
    }

    Write-Host "Atualizando dependências..." -ForegroundColor Cyan
    & .\venv\Scripts\python.exe -m pip install -e . --quiet

    $newVer = & .\venv\Scripts\python.exe -c "import src; print(src.__version__)" 2>&1
    Write-Host ""
    Write-Host "Atualização concluída! Versão: $newVer" -ForegroundColor Green
    Write-Host "Execute: python gui_app.py" -ForegroundColor Cyan
    exit 0
}

# ── Modo de INSTALAÇÃO NOVA ───────────────────────────────────────────
Write-Host ""
Write-Host "=== IonFlow Pipeline — INSTALAÇÃO ===" -ForegroundColor Cyan
Write-Host ""

# Se nenhum parâmetro, perguntar
if (-not $Ciclagem -and -not $EIS) {
    Write-Host "Escolha o pipeline para instalar:"
    Write-Host "  1 - Ciclagem"
    Write-Host "  2 - EIS"
    Write-Host "  3 - Ambos (recomendado)"
    $choice = Read-Host "Digite o número da opção"
    if ($choice -eq "1") { $Ciclagem = $true }
    elseif ($choice -eq "2") { $EIS = $true }
    else { $Ciclagem = $true; $EIS = $true }
}

Install-Python311

# Desbloquear arquivos se vieram de outro computador (Windows MOTW)
Write-Host "Desbloqueando arquivos..." -ForegroundColor Cyan
Get-ChildItem -Path . -Recurse -File | Unblock-File -ErrorAction SilentlyContinue

Create-Venv
Install-Requirements
Setup-Pipelines

$ver = & .\venv\Scripts\python.exe -c "import src; print(src.__version__)" 2>&1
Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host " Instalação concluída!  IonFlow Pipeline v$ver" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Para usar:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1   # ativar ambiente"
Write-Host "  python gui_app.py              # abrir GUI"
Write-Host ""
Write-Host "Para atualizar no futuro:" -ForegroundColor Cyan
Write-Host "  .\install_pipeline.ps1 -Update"
Write-Host ""

