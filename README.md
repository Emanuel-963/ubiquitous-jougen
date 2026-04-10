# IonFlow Pipeline

Toolkit completo para análise de **Espectroscopia de Impedância Eletroquímica (EIS)**, com GUI interativa, DRT, ciclagem galvanostática e 17 abas de visualização.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Funcionalidades

| Módulo | Descrição |
|---|---|
| **Pipeline EIS** | Carrega dados brutos, extrai métricas físicas (Rs, Rp, CPE), PCA, ranking e classificação automática |
| **Pipeline DRT** | Distribution of Relaxation Times via regularização de Tikhonov |
| **Pipeline Ciclagem** | Energia e potência por ciclo a partir de curvas galvanostáticas |
| **GUI Interativa** | 17 abas com gráficos interativos (hover, zoom, pan) via customtkinter + matplotlib |

### 17 Abas da GUI

1. **Tabela de Resultados** — métricas físicas extraídas
2. **Rank vs Retenção** — scatter interativo
3. **Nyquist** — -Z'' vs Z' com hover por amostra
4. **Bode** — |Z| e fase vs frequência
5. **PCA 2D** — projeção com cores por subclasse
6. **PCA Retenção** — PCA colorido por métrica contínua
7. **Correlação** — heatmap Spearman
8. **Séries** — valor vs amostra ordenado
9. **Energia × Potência** — evolução por ciclo
10. **Ragone** — log Energia vs log Potência
11. **Energia vs Ciclo** — capacitância/energia por ciclo
12. **Retenção vs Ciclo** — curva de retenção percentual
13. **Heatmap |Z|** — impedância por frequência e amostra
14. **Box-plot** — distribuição comparativa de métricas
15. **Radar** — perfil multi-métrica por amostra
16. **DRT** — espectro g(τ) por amostra
17. **DRT × EIS** — overlay DRT + Nyquist

## 📁 Estrutura do Projeto

```
eis_analytics/
├── gui_app.py              # GUI principal (17 abas)
├── main.py                 # Pipeline EIS
├── main_cycling.py         # Pipeline ciclagem
├── main_drt.py             # Pipeline DRT
├── build_exe.py            # Build do executável
├── pyproject.toml          # Metadados e dependências
├── src/                    # Módulos de análise
│   ├── loader.py           # Carregamento de dados EIS
│   ├── preprocessing.py    # Limpeza e filtragem
│   ├── physics_metrics.py  # Extração de Rs, Rp, CPE, etc.
│   ├── pca_analysis.py     # PCA e biplot
│   ├── ranking.py          # Classificação e ranking
│   ├── stability.py        # Agrupamento por amostra
│   ├── visualization.py    # Gráficos estáticos
│   ├── eis_plots.py        # 8 gráficos interativos
│   ├── drt_analysis.py     # Cálculo DRT
│   ├── drt_visualization.py# Visualização DRT
│   ├── circuit_fitting.py  # Ajuste de circuitos
│   ├── cpe_fit.py          # Ajuste CPE
│   ├── cycling_loader.py   # Loader de ciclagem
│   ├── cycling_calculator.py # Cálculo energia/potência
│   └── cycling_plotter.py  # Gráficos de ciclagem
├── data/
│   ├── raw/                # Arquivos EIS (.txt)
│   ├── processed/          # Dados de ciclagem
│   └── ionflowMarca.png    # Logo
├── themes/
│   └── ionflow.json        # Tema visual da GUI
├── tests/                  # 44 testes unitários
└── outputs/                # Tabelas, figuras, Excel
```

## 🚀 Instalação

### Via pip (modo desenvolvedor)

```bash
python -m venv venv
# Windows:
venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

pip install -e ".[dev]"
```

### Executável standalone (Windows)

```bash
pip install -e ".[build]"
python build_exe.py            # pasta dist/IonFlow_Pipeline/
python build_exe.py --onefile  # arquivo único .exe
```

O executável fica em `dist/IonFlow_Pipeline/IonFlow_Pipeline.exe`.

## ▶️ Uso

### GUI

```bash
ionflow                 # via entry-point (após pip install)
python gui_app.py       # direto
```

### Linha de comando

```bash
python main.py          # Pipeline EIS completo
python main_cycling.py  # Pipeline ciclagem
python main_drt.py      # Pipeline DRT
```

Coloque os arquivos `.txt` de EIS em `data/raw/` e de ciclagem em `data/processed/`.

## 🧪 Desenvolvimento

```bash
pip install -e ".[dev]"
pytest -q                # 44 testes
black . && isort .       # formatação
flake8 .                 # linting
mypy src                 # type checking
```

## 📊 Metodologia

- **Capacitância efetiva** dependente da frequência
- **Rs** (resistência série) e **Rp** (resistência de polarização)
- **CPE** (Constant Phase Element) fitting
- **DRT** via regularização de Tikhonov
- **Energia e potência** por ciclo (Wh/kg, W/kg)
- **PCA** interpretado fisicamente
- **Classificação automática**: Bateria, Supercapacitor, Célula eletroquímica, Célula fotoeletroquímica

## 📄 Licença

MIT — veja [LICENSE](LICENSE).

## Contexto Acadêmico
Projeto desenvolvido para análise de materiais eletroquímicos em contexto de iniciação científica.
## How to present this work ✅
- Short demo: run `python scripts/regenerate_figures.py` to generate an example PCA figure in `outputs/figures`.
- Key slides: Motivation, Methods (metrics: Rs, Rp, C_eff, Tau), Results (Nyquist & PCA), Reproducibility.
- One-pager: see `docs/ONE_PAGER.md` for a 1-page summary and demo steps.
- Notes: aim for a 5–10 minute demo focusing on motivation, a single clear result, and how to reproduce it locally.
## Licença
Este projeto está licenciado sob a Licença MIT — veja o arquivo `LICENSE` para detalhes.

