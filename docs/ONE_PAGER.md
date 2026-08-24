# IonFlow Pipeline v0.5.0 — One-page Summary

## What it is

IonFlow Pipeline is a professional, lab-grade analytics platform for **Electrochemical
Impedance Spectroscopy (EIS)**, **galvanostatic cycling**, and **Distribution of
Relaxation Times (DRT)** analysis. It features an AI-powered interpretation agent,
PDF reporting, parallel processing, and an interactive GUI — all in a single installable
desktop application.

## Key Features (v0.5.0)

| Category | Capabilities |
|----------|-------------|
| **EIS Analysis** | 7 equivalent circuit models (extensible registry), auto-selection via ML classifier, Monte Carlo uncertainty, Kramers-Kronig validation |
| **Cycling** | Ragone plots with technology reference zones, gap analysis vs targets (300 Wh/kg, 3000 W/kg), retention metrics |
| **DRT** | Tikhonov regularisation, peak detection, multi-sample overlay |
| **AI Agent** | 50+ electrochemical rules, inference engine, performance predictor, process advisor, optional LLM enrichment (OpenAI/Ollama) |
| **Visualisation** | Nyquist, Bode, PCA 2D/3D, correlation heatmaps, production heatmaps (material × synthesis), boxplots |
| **Reporting** | Automated PDF with cover, EIS/Cycling/DRT/AI sections, images, tables |
| **GUI** | Workspace V3, guided project wizard with objective presets, contextual Ribbon, Command Palette (`Ctrl+K`), scrollable Inspector, 3 languages |
| **CLI** | `ionflow-cli eis / cycling / drt / analyze / validate / config` |
| **Quality** | Automated tests, GitHub Actions CI, structured logging, determinate GUI progress with stage and percentage |

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

1. Launch the GUI and open **Iniciar wizard de projeto** in the Workspace
2. Choose a preset: quick triage, EIS diagnosis, cycling performance, complete characterization, DRT validation, or publishable report
3. Place your EIS `.csv` files in `data/raw/` and cycling files in `data/processed/`
4. Review the preset and execute the suggested pipeline
5. Follow the percentage and current stage in the GUI; detailed messages remain in Logs
6. The AI agent interprets results and recommends process improvements
7. Export a complete PDF report with one click

## Architecture

- **35+ Python modules** with typed dataclasses and structured logging
- **`src/ai/`** — Knowledge base (50+ rules), inference engine, performance predictor, process advisor, LLM adapter
- **`src/gui/`** — MVC pattern with modular tab system
- **Windows distribution** — Inno Setup installer with silent in-place updates from GitHub Releases
- **`src/config.py`** — Single `PipelineConfig` dataclass (zero magic numbers)

## Links

- Repository: https://github.com/Emanuel-963/ubiquitous-jougen
- License: MIT

