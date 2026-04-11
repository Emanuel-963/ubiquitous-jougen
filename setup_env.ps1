Write-Host "======================================"
Write-Host " Setting up EIS Analytics Environment "
Write-Host "======================================"

# Verifica Python
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python não encontrado. Instale Python >= 3.9."
    exit 1
}

# Cria ambiente virtual
if (!(Test-Path "venv")) {
    Write-Host "Criando ambiente virtual..."
    python -m venv venv
}

# Ativa ambiente virtual
Write-Host "Ativando ambiente virtual..."
.\venv\Scripts\Activate.ps1

# Atualiza pip
python -m pip install --upgrade pip

# Instala dependências
pip install numpy pandas matplotlib scikit-learn scipy

Write-Host "======================================"
Write-Host " Ambiente configurado com sucesso "
Write-Host "======================================"
