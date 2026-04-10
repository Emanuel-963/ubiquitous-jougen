#!/usr/bin/env powershell
<#+
.SYNOPSIS
    Instalador automatizado para o pipeline de Ciclagem e/ou EIS.
.DESCRIPTION
    - Cria ambiente virtual isolado
    - Baixa e instala Python 3.11 se necessário
    - Instala todas as dependências do requirements.txt
    - Permite escolher pipeline de Ciclagem, EIS ou ambos
    - Cria as pastas necessárias para cada pipeline
    - Configura variáveis de ambiente/PATH se necessário
    - Organiza tudo em estrutura limpa
#>

param(
    [switch]$Ciclagem,
    [switch]$EIS
)

# Função para baixar e instalar Python 3.11
function Install-Python311 {
    $pythonUrl = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
    $installer = "$env:TEMP\python-3.11.7-amd64.exe"
    if (-not (Get-Command python3.11 -ErrorAction SilentlyContinue)) {
        Write-Host "Baixando Python 3.11..."
        Invoke-WebRequest -Uri $pythonUrl -OutFile $installer
        Write-Host "Instalando Python 3.11..."
        Start-Process -Wait -FilePath $installer -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1 Include_test=0'
        Remove-Item $installer
    } else {
        Write-Host "Python 3.11 já instalado."
    }
}

# Função para criar ambiente virtual
function Create-Venv {
    $venvPath = "venv"
    if (-not (Test-Path $venvPath)) {
        Write-Host "Criando ambiente virtual..."
        python3.11 -m venv $venvPath
    } else {
        Write-Host "Ambiente virtual já existe."
    }
}

# Função para instalar dependências
function Install-Requirements {
    $venvPython = ".\venv\Scripts\python.exe"
    Write-Host "Instalando dependências do requirements.txt..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
    if (Test-Path "requirements-dev.txt") {
        & $venvPython -m pip install -r requirements-dev.txt
    }
}

# Função para criar pastas dos pipelines
function Setup-Pipelines {
    Write-Host "Configurando pastas dos pipelines..."
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
}

# Função para configurar variáveis de ambiente
function Setup-EnvVars {
    $venvPath = (Resolve-Path "venv").Path
    [System.Environment]::SetEnvironmentVariable("VIRTUAL_ENV", $venvPath, [System.EnvironmentVariableTarget]::User)
    $venvBin = Join-Path $venvPath "Scripts"
    $oldPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
    if ($oldPath -notlike "*$venvBin*") {
        [System.Environment]::SetEnvironmentVariable("Path", "$venvBin;$oldPath", [System.EnvironmentVariableTarget]::User)
    }
}

# Menu de seleção
Write-Host "--- Instalador do Pipeline ---"
Write-Host "Escolha o pipeline para instalar:"
Write-Host "1 - Ciclagem"
Write-Host "2 - EIS"
Write-Host "3 - Ambos"
$choice = Read-Host "Digite o número da opção desejada"

if ($choice -eq "1") { $Ciclagem = $true }
elseif ($choice -eq "2") { $EIS = $true }
elseif ($choice -eq "3") { $Ciclagem = $true; $EIS = $true }
else { Write-Host "Opção inválida."; exit 1 }

Install-Python311
Create-Venv
Install-Requirements
if ($Ciclagem -or $EIS) { Setup-Pipelines }
Setup-EnvVars

Write-Host "\nInstalação concluída! Ative o ambiente com:"
Write-Host ".\venv\Scripts\Activate.ps1"
Write-Host "Depois rode: python main_cycling.py (ou o script do pipeline EIS)"
