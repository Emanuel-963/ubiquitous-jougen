# IonFlow Pipeline

**Plataforma profissional de análise eletroquímica** — EIS, ciclagem galvanostática, DRT — com agente IA, relatórios PDF, CLI e GUI interativa.

[![CI](https://github.com/Emanuel-963/ubiquitous-jougen/actions/workflows/ci.yml/badge.svg)](https://github.com/Emanuel-963/ubiquitous-jougen/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.2.0-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-1819%20passing-brightgreen)

---

## ⚡ Quick Start

```bash
# 1. Instalar
python -m venv venv && venv\Scripts\Activate.ps1
pip install -e .

# 2. GUI
python gui_app.py

# 3. CLI
ionflow-cli eis --data-dir data/raw --output outputs/
ionflow-cli analyze --all --ai --export-pdf report.pdf
```

---

## ✨ Funcionalidades (v0.2.0)

| Módulo | Descrição |
|---|---|
| **Pipeline EIS** | 7 circuitos equivalentes, ML shortlist, Monte Carlo, Kramers-Kronig |
| **Pipeline DRT** | Tikhonov regularisation, detecção de picos, overlay multi-amostra |
| **Pipeline Ciclagem** | Ragone com zonas de referência, gap analysis vs targets |
| **🤖 Agente IA** | 50+ regras eletroquímicas, motor de inferência, predição, process advisor |
| **📄 PDF Reports** | Relatório automático com capa, secções EIS/Ciclagem/DRT/IA |
| **CLI** | `ionflow-cli eis / cycling / drt / analyze / validate / config` |
| **GUI** | MVC, 6 módulos de abas, atalhos teclado, 3 idiomas (PT/EN/ES) |
| **Batch** | Processamento paralelo com `ProcessPoolExecutor` |
| **i18n** | 3 idiomas com troca em tempo real |

---

## 🔬 Para Pesquisadores

### Fluxo típico de análise

1. **Preparar dados** — Colocar ficheiros EIS (`.csv`/`.txt`) em `data/raw/`
2. **Executar** — `python gui_app.py` ou `ionflow-cli eis --data-dir data/raw`
3. **Pipeline automático:**
   - Carregamento → Validação (Kramers-Kronig) → Fitting (7 circuitos × multi-seed)
   - Ranking por BIC → PCA → Heatmaps de produção → Classificação
4. **Análise IA** — Clique em "🤖 Análise IA" para obter:
   - Resumo executivo dos resultados
   - Anomalias detectadas (ex: "Rs do Na₂SO₄ é 4.5× maior que H₂SO₄")
   - Recomendações priorizadas (ex: "Polir eletrodo para reduzir Rs")
   - Previsões: energia e retenção estimadas
5. **Exportar** — PDF com 1 clique, ou `ionflow-cli analyze --export-pdf report.pdf`

### Métricas extraídas

- **Rs** (resistência série), **Rp** (resistência de polarização)
- **CPE** (n, Q), **Sigma** (Warburg), **Tau** (constante de tempo)
- **Capacitância efetiva**, **Energia armazenada**, **Dispersion Index**
- **Incerteza:** Monte Carlo (N=100) + Bootstrap → intervalos de confiança 95%

---

## 🤖 Agente IA — Exemplos

```
📊 Resumo Executivo
"A amostra Nb₂/H₂SO₄ apresenta Rs=2.66Ω (baixo) e n=0.77 (rugosidade moderada).
O circuito Randles-CPE-W foi selecionado com 78% de confiança (BIC=45.2)."

⚠️ Anomalias
• Rs do Na₂SO₄ é 4.5× maior que amostras em H₂SO₄
• Rp convergiu para o limite superior — possível não-convergência

💡 Recomendações
1. [ALTA] Polir eletrodo ou usar cola de prata para reduzir Rs
2. [MÉDIA] Expandir faixa de frequência para 10 mHz–1 MHz
3. [MÉDIA] H₂SO₄ proporciona Rs 78% menor → priorizar como eletrólito

🔮 Previsões
• Energia estimada: 12 ± 3 μJ
• Retenção estimada: 85 ± 8%
```

---

## 📁 Estrutura do Projeto

```
eis_analytics/
├── gui_app.py                # GUI principal
├── main.py / main_cycling.py / main_drt.py
├── build_exe.py              # Build PyInstaller
├── pyproject.toml            # Metadados + dependências
├── src/                      # 35+ módulos
│   ├── config.py             # PipelineConfig centralizado
│   ├── models.py             # Dataclasses tipadas (EISResult, etc.)
│   ├── validation.py         # Validação de entrada + KK
│   ├── kramers_kronig.py     # Teste Kramers-Kronig (Boukamp)
│   ├── circuit_registry.py   # 7 circuitos extensíveis
│   ├── circuit_composer.py   # Recombinação automática
│   ├── ml_circuit_selector.py# ML shortlist (RandomForest)
│   ├── fitting_diagnostics.py# Diagnóstico visual 🟢🟡🔴
│   ├── fitting_report.py     # Feedback textual
│   ├── uncertainty.py        # Monte Carlo + Bootstrap
│   ├── report_generator.py   # PDF automático (fpdf2)
│   ├── batch_processor.py    # Processamento paralelo
│   ├── cli.py                # CLI profissional
│   ├── i18n.py               # 3 idiomas (PT/EN/ES)
│   ├── ai/                   # Agente IA
│   │   ├── knowledge_base.py # 50+ regras eletroquímicas
│   │   ├── inference_engine.py
│   │   ├── performance_predictor.py
│   │   ├── process_advisor.py
│   │   └── llm_adapter.py    # LLM opcional (OpenAI/Ollama)
│   └── gui/                  # GUI modular (MVC)
│       ├── controller.py / models.py / main_window.py
│       ├── shortcuts.py / widgets.py
│       └── tabs/ (eis, cycling, drt, advanced, tables, ai_panel)
├── data/knowledge/           # Regras eletroquímicas (JSON)
├── tests/                    # 1782 testes automatizados
├── themes/                   # Tema visual da GUI
└── outputs/                  # Tabelas, figuras, Excel, PDF
```

---

## 🚀 Instalação

### Via pip (modo desenvolvedor)

```bash
python -m venv venv
venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate  # Linux/Mac
pip install -e ".[dev]"
```

### Executável standalone (Windows)

```bash
pip install -e ".[build]"
python build_exe.py
# → dist/IonFlow_Pipeline/IonFlow_Pipeline.exe
```

### Instalador Windows (Inno Setup)

```bash
iscc installer/ionflow_setup.iss
# → dist/installer/IonFlow_Pipeline_Setup_0.2.0.exe
```

---

## ▶️ Uso

### GUI

```bash
ionflow                 # via entry-point
python gui_app.py       # direto
```

### CLI

```bash
ionflow-cli eis --data-dir data/raw --output outputs/
ionflow-cli cycling --data-dir data/processed --scan-rate 0.1
ionflow-cli drt --data-dir data/raw --lambda 1e-3
ionflow-cli analyze --all --ai --export-pdf report.pdf
ionflow-cli validate --data-dir data/raw
ionflow-cli config --init
```

### Scripts de pipeline

```bash
python main.py          # Pipeline EIS completo
python main_cycling.py  # Pipeline ciclagem
python main_drt.py      # Pipeline DRT
```

---

## 🧪 Desenvolvimento

```bash
pip install -e ".[dev]"
pytest -q                # 1782 testes
black . && isort .       # formatação
flake8 .                 # linting
mypy src                 # type checking
```

---

## 📊 Métricas v0.1.0 → v0.2.0

| Métrica | v0.1.0 | v0.2.0 |
|---------|--------|--------|
| Módulos Python | 18 | **35+** |
| Circuitos | 3 | **7** (extensível) |
| Testes | ~70% cov | **1782 testes** |
| Idiomas | 2 (PT/EN) | **3** (+ ES) |
| Agente IA | ❌ | ✅ Rule-based + ML + LLM |
| PDF Reports | ❌ | ✅ Automático |
| CLI | ❌ | ✅ Completa |
| Monte Carlo | ❌ | ✅ N=100 + Bootstrap |
| Kramers-Kronig | ❌ | ✅ Validação automática |
| Batch Processing | ❌ | ✅ Paralelo |

---

## 📄 Licença

MIT — veja [LICENSE](LICENSE).

## 🎓 Contexto Acadêmico

Projeto desenvolvido para análise de materiais eletroquímicos em contexto de investigação científica.

## 📚 Documentação

- [CHANGELOG](CHANGELOG.md) — Histórico de alterações
- [ONE_PAGER](docs/ONE_PAGER.md) — Resumo de 1 página
- [PRESENTATION](docs/PRESENTATION.md) — Guia de apresentação
- [UPGRADE_PLAN](docs/UPGRADE_PLAN_v0.2.0.md) — Plano de 30 dias
- [Tutoriais](tutoriais/) — Passo a passo

