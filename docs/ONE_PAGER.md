# IonFlow Pipeline v0.2.0 — One-page Summary

## What it is

IonFlow Pipeline is a professional, lab-grade analytics platform for **Electrochemical
Impedance Spectroscopy (EIS)**, **galvanostatic cycling**, and **Distribution of
Relaxation Times (DRT)** analysis. It features an AI-powered interpretation agent,
PDF reporting, parallel processing, and an interactive GUI — all in a single installable
desktop application.

## Key Features (v0.2.0)

| Category | Capabilities |
|----------|-------------|
| **EIS Analysis** | 7 equivalent circuit models (extensible registry), auto-selection via ML classifier, Monte Carlo uncertainty, Kramers-Kronig validation |
| **Cycling** | Ragone plots with technology reference zones, gap analysis vs targets (300 Wh/kg, 3000 W/kg), retention metrics |
| **DRT** | Tikhonov regularisation, peak detection, multi-sample overlay |
| **AI Agent** | 50+ electrochemical rules, inference engine, performance predictor, process advisor, optional LLM enrichment (OpenAI/Ollama) |
| **Visualisation** | Nyquist, Bode, PCA 2D/3D, correlation heatmaps, production heatmaps (material × synthesis), boxplots |
| **Reporting** | Automated PDF with cover, EIS/Cycling/DRT/AI sections, images, tables |
| **GUI** | MVC architecture, 6 tab modules, keyboard shortcuts, 3 languages (PT/EN/ES), accessibility settings |
| **CLI** | `ionflow-cli eis / cycling / drt / analyze / validate / config` |
| **Quality** | 1782 automated tests, GitHub Actions CI, structured logging |

## Quick Start

```bash
# Install
pip install -e .

# GUI
ionflow

# CLI
ionflow-cli eis --data-dir data/raw --output outputs/
ionflow-cli analyze --all --ai --export-pdf report.pdf

# Or run directly
python gui_app.py
```

## For Researchers

1. Place your EIS `.csv` files in `data/raw/`
2. Launch the GUI or CLI
3. The pipeline automatically: loads → validates (Kramers-Kronig) → fits 7 circuits → ranks by BIC → runs PCA → generates heatmaps
4. The AI agent interprets results and recommends process improvements
5. Export a complete PDF report with one click

## Architecture

- **35+ Python modules** with typed dataclasses and structured logging
- **`src/ai/`** — Knowledge base (50+ rules), inference engine, performance predictor, process advisor, LLM adapter
- **`src/gui/`** — MVC pattern with modular tab system
- **`src/config.py`** — Single `PipelineConfig` dataclass (zero magic numbers)

## Links

- Repository: https://github.com/Emanuel-963/ubiquitous-jougen
- License: MIT

