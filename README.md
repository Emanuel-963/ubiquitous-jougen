# EIS Analytics Framework

Framework para análise automática de dados de Espectroscopia de Impedância Eletroquímica (EIS), com extração de métricas físicas, análise multivariada (PCA) e inferência da aplicação tecnológica mais adequada para cada amostra.

## Objetivo
- Processar qualquer número de amostras de EIS
- Extrair parâmetros físicos sem forçar circuitos equivalentes
- Comparar materiais de forma objetiva
- Classificar aplicação provável:
  - Bateria
  - Supercapacitor
  - Célula eletroquímica genérica
  - Célula fotoeletroquímica / fotovoltaica

## Estrutura do Projeto
eis_analytics/
├── data/
│   ├── raw/
│   └── processed/
├── src/
├── outputs/
│   ├── tables/
│   └── figures/
└── main.py

## Instalação
Execute o script de configuração do ambiente:

bash setup_env.sh
source venv/bin/activate

## Execução
Coloque os arquivos de EIS em data/raw/ e execute:

python main.py

## Desenvolvimento

- Criar e ativar ambiente virtual (Windows):
  - `python -m venv venv && venv\Scripts\Activate.ps1`
- Instalar dependências de desenvolvimento: `pip install -r requirements-dev.txt`
- Instalar pacote em modo editável: `pip install -e .`
- Rodar testes: `pytest -q`
- Rodar linters/formatters: `black . && isort . && flake8 .`
- Instalar hooks do pre-commit: `pre-commit install`

> Dica: para desenvolvimento local recomendamos usar **Python 3.11** para garantir compatibilidade com as ferramentas de linting (pre-commit irá criar ambientes com Python 3.11 quando possível).

## Saídas
- Tabela com propriedades físicas extraídas
- Scores de PCA
- Classificação automática de aplicação

## Metodologia
- Capacitância efetiva dependente da frequência
- Resistência série (Rs) e resistência de polarização (Rp)
- Energia acumulada
- Constante de tempo dominante
- PCA interpretado fisicamente

## Contexto Acadêmico
Projeto desenvolvido para análise de materiais eletroquímicos em contexto de iniciação científica.
## How to present this work ✅
- Short demo: run `python scripts/regenerate_figures.py` to generate an example PCA figure in `outputs/figures`.
- Key slides: Motivation, Methods (metrics: Rs, Rp, C_eff, Tau), Results (Nyquist & PCA), Reproducibility.
- One-pager: see `docs/ONE_PAGER.md` for a 1-page summary and demo steps.
- Notes: aim for a 5–10 minute demo focusing on motivation, a single clear result, and how to reproduce it locally.
## Licença
Este projeto está licenciado sob a Licença MIT — veja o arquivo `LICENSE` para detalhes.

