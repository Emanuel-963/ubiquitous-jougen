# Changelog

All notable changes to the IonFlow Pipeline are documented here.

## [0.3.0] — 2026-04-28

### Highlights

**IonFlow Pipeline v0.3.0** expands the platform with native instrument parsers, scientific
export formats, multi-sample comparative analysis, generative AI, persistent SQLite storage,
and a Streamlit web dashboard — completing 7 planned development phases.

- **50+ Python modules**
- **220+ new automated tests** (7 phases)
- **4 native parsers**: Gamry, BioLogic, Autolab, Zahner
- **4 scientific export formats**: ZView, LaTeX booktabs, OriginPro, MEISP
- **Comparative analysis** with Health Score (0–10) and automatic PCA
- **Generative AI**: OpenAI (gpt-4o) + Ollama (local) via unified LLMAdapter
- **Settings panel** with persistent JSON config + GitHub auto-update
- **SQLite backend** (7 tables, migrations) + Streamlit dashboard (7 pages)

---

### Added — Phase 2: Native Instrument Parsers

- **`src/parsers/gamry.py`** — Gamry `.dta` parser
- **`src/parsers/biologic.py`** — BioLogic `.mpr` / `.mpt` parser
- **`src/parsers/autolab.py`** — Metrohm Autolab `.csv` parser
- **`src/parsers/zahner.py`** — Zahner `.isc` parser
- **`src/parsers/__init__.py`** — Unified `parse_eis_file()` with auto-detection

### Added — Phase 3: Scientific Export

- **`src/export/zview.py`** — ZView/ZPlot `.z` format
- **`src/export/latex_export.py`** — LaTeX booktabs `.tex` tables
- **`src/export/origin.py`** — OriginPro `.csv` with metadata header
- **`src/export/meisp.py`** — MEISP `.txt` format

### Added — Phase 4: Comparative Analysis

- **`src/comparison/`** — Multi-sample overlay (Nyquist + Bode + DRT)
- **Health Score** — Weighted 0–10 score per sample (Rs, Rp, BIC, KK, ML confidence)
- **PCA** — Automatic k-means clustering with 2D visualisation
- GUI tab **📊 Comparar** with exportable ranking table

### Added — Phase 5: Generative AI

- **`src/ai/llm_adapter.py`** — `LLMAdapter` (OpenAI + Ollama + NullAdapter)
- Automatic scientific narrative in Portuguese, embedded in PDF reports

### Added — Phase 6: Settings + Auto-Update

- **Settings panel** (`⚙️ Configurações`) — persistent JSON config
- **`src/updater.py`** — GitHub release checker + in-GUI download

### Added — Phase 7: SQLite Backend + Streamlit Dashboard

- **`src/db/schema.py`** — 7-table SQLite schema with WAL mode and FK cascade
- **`src/db/migrations.py`** — Versioned, idempotent migrations
- **`src/db/repository.py`** — `IonFlowRepository` CRUD (samples, EIS, DRT, cycling, params)
- **`src/db/feature_store_v2.py`** — `FeatureStoreV2` with similarity search and ML history
- **`dashboard.py`** — Streamlit multipage app (Overview, Upload, EIS, DRT, Cycling, History, AI)
- Launch: `python -m streamlit run dashboard.py` → `http://localhost:8501`

### Added — Tutorials

- `tutoriais/16_novidades_v0.3.0.txt` — Overview of all phases + upgrade guide
- `tutoriais/17_dashboard_streamlit.txt` — Streamlit dashboard guide
- `tutoriais/18_base_dados_sqlite.txt` — IonFlowRepository + FeatureStoreV2 with examples
- `tutoriais/19_importacao_potenciostatos.txt` — Gamry/BioLogic/Autolab/Zahner parsers
- `tutoriais/20_exportacao_cientifica.txt` — ZView/LaTeX/Origin/MEISP export
- `tutoriais/21_analise_comparativa.txt` — Multi-sample comparison + Health Score + PCA
- `tutoriais/22_configuracoes_e_autoupdate.txt` — Settings panel + auto-update + env vars

---

## [0.2.0] — 2026-04-16

### Highlights

**IonFlow Pipeline v0.2.0** is a major upgrade transforming the project from a simple EIS
analysis script into a professional, lab-grade analytics platform with AI-assisted
interpretation, PDF reporting, parallel processing, and a fully modular GUI.

- **35+ Python modules** (up from 18)
- **1782 automated tests** passing
- **7 equivalent circuit models** (up from 3)
- **3 languages** (PT, EN, ES)
- **AI agent** with rule-based inference, performance prediction, and process advisor
- **CLI, PDF reports, batch processing, Monte Carlo uncertainty**

---

### Added — Infrastructure (Week 1)

- **`src/config.py`** — Centralised `PipelineConfig` dataclass; JSON serialisation, defaults,
  validation. All magic numbers eliminated. *(Day 1)*
- **`src/models.py`** — Typed result dataclasses: `EISResult`, `CyclingResult`,
  `DRTPipelineResult`, `PCAResult`. *(Day 2)*
- **`src/logger.py`** — Structured logging with rotating file handler
  (`logs/ionflow.log`), coloured console, GUI queue handler. *(Day 3)*
- **`src/validation.py`** — Input validators: `validate_eis_dataframe`,
  `validate_cycling_dataframe`, `validate_frequency_range`,
  `validate_impedance_quality`, `validate_eis_full`. *(Day 3)*
- **`src/circuit_registry.py`** — `CircuitRegistry` pattern with 7 circuits:
  Randles-CPE, Randles-CPE-W, Double-ZARC, Coating-CPE, Warburg-Finite,
  ZARC-ZARC-W, Simple-RC. Each includes physical meanings and typical systems. *(Day 4)*
- **`src/feature_store.py`** — `FeatureStore` + `FittingHistory` for ML training data
  persistence; `similar_samples()` via Euclidean distance. *(Day 5)*
- **`src/ml_circuit_selector.py`** — `CircuitMLSelector` with RandomForest classifier;
  fallback to heuristic when < 30 samples; textual feedback. *(Day 6)*
- **`tests/test_integration.py`** — End-to-end pipeline tests. *(Day 7)*
- **`.github/workflows/ci.yml`** — GitHub Actions CI: Python 3.11/3.12 matrix,
  pytest, flake8, coverage. *(Day 7)*

### Added — Advanced ML & Fitting (Week 2)

- **`src/circuit_composer.py`** — `CircuitComposer` auto-generates circuit candidates
  from fundamental blocks (R, C, CPE, W, W_finite, L, ZARC);
  series/parallel topologies; `auto_select()` with BIC ranking. *(Day 8)*
- **`src/fitting_diagnostics.py`** — Rich visual diagnostics: Nyquist overlay, Bode,
  residual analysis, parameter confidence bars, model comparison; traffic-light
  quality indicators (🟢🟡🔴). *(Day 9)*
- **`src/fitting_report.py`** — `FittingReportGenerator` with textual interpretation
  of every fitted parameter, circuit justification, quality assessment, and
  comparison with historical samples. *(Day 10)*
- **`src/uncertainty.py`** — `UncertaintyAnalyzer`: Monte Carlo error propagation
  (N=100) and bootstrap residual resampling; 95% confidence intervals. *(Day 11)*
- **`src/kramers_kronig.py`** — `KramersKronigValidator` (Boukamp method): Voigt
  element fitting, residual classification (< 1% excellent, < 5% acceptable),
  plot generation. *(Day 12)*

### Added — GUI Refactoring (Week 2)

- **`src/gui/`** package with MVC architecture: *(Day 13–14)*
  - `controller.py` — `PipelineController` (orchestration logic)
  - `models.py` — `AppState` dataclass (view-model)
  - `main_window.py` — `MainWindow` (pure layout + bindings)
  - `widgets.py` — Reusable widgets (`StyledOptionMenu`, `FilterableTable`, etc.)
  - `tabs/` — Modular chart tabs: `eis_charts`, `cycling_charts`, `drt_charts`,
    `advanced_charts`, `tables`, `ai_panel`

### Added — AI Agent (Week 3)

- **`src/ai/knowledge_base.py`** — 50+ electrochemical rules covering Rs, Rp, CPE-n,
  Warburg σ, cycling retention, DRT peaks, cross-correlations. *(Day 15)*
- **`src/ai/inference_engine.py`** — `InferenceEngine` producing `AnalysisReport`
  with findings, anomalies, recommendations, quality score, and executive
  summary. Cross-pipeline analysis (EIS + Cycling + DRT). *(Day 16)*
- **`src/ai/performance_predictor.py`** — `PerformancePredictor`:
  `predict_cycling_from_eis()`, `predict_degradation()`,
  `recommend_improvements()`. Ridge/RF regression on FeatureStore. *(Day 17)*
- **`src/ai/process_advisor.py`** — `ProcessAdvisor`:
  `analyze_material_system()` → material assessment, best conditions,
  bottleneck analysis, production recommendations, next experiments. *(Day 18)*
- **`src/gui/tabs/ai_panel.py`** — GUI "🤖 Análise IA" tab with executive summary,
  anomalies, recommendations, predictions, and PDF export. *(Day 19)*
- **`src/ai/llm_adapter.py`** — Optional LLM integration layer:
  `OpenAIAdapter`, `OllamaAdapter`, `NullAdapter` (offline fallback). *(Day 20)*

### Added — Professional Polish (Week 4)

- **`src/cli.py`** — Full CLI: `ionflow-cli eis`, `cycling`, `drt`, `analyze`,
  `config --init`, `validate`; tqdm progress, JSON output, return codes. *(Day 22)*
- **`src/report_generator.py`** — PDF report generation with fpdf2: cover page,
  EIS/Cycling/DRT/Correlation/AI sections, images, tables. *(Day 23)*
- **`src/batch_processor.py`** — `BatchProcessor` + `ParallelFitter`:
  `ProcessPoolExecutor`, cancellation support, memory guard. *(Day 24)*
- **`src/i18n.py`** + `src/i18n_strings/{pt,en,es}.json` — Full i18n with
  3 languages, section-based keys, runtime switching. *(Day 25)*
- **`src/gui/shortcuts.py`** — Keyboard shortcuts (Ctrl+1/2/3, F5, Escape, etc.),
  `StatusBarState`, `TooltipRegistry`, `AccessibilitySettings`. *(Day 26)*

### Added — Testing & Quality

- **57 edge-case tests** in `test_day28_bugbash.py` covering preprocessing,
  loader, stability, metadata, cycling, eis_plots, visualization. *(Day 28)*
- **30 production visualisation tests** in `test_production_viz.py`. *(Mid-plan)*
- **Total: 1782 tests**, 0 failures.

### Changed

- **`src/preprocessing.py`** — Fixed `SettingWithCopyWarning` by adding `.copy()`
  after `dropna()` and frequency filter. *(Day 28)*
- **`src/metadata.py`** — Added `extract_material_type()`, `extract_synthesis_process()`,
  `extract_full_metadata()` for production variables (Nb2/Nb4, Prisca). *(Mid-plan)*
- **`src/visualization.py`** — Added `production_heatmap()` with z-score normalisation
  and RdYlGn colormap. *(Mid-plan)*
- **`src/eis_plots.py`** — `plot_ragone()` now supports target markers and reference
  technology zones; added `RagoneGapResult` and `ragone_gap_analysis()`. *(Mid-plan)*
- **`main.py`** — Integrated production heatmap and metadata extraction. *(Mid-plan)*
- **`pyproject.toml`** — Added `fpdf2>=2.7`, `tqdm>=4.65`, `requests>=2.28`. *(Day 29)*
- **`build_exe.py`** — 35+ hidden imports, `src/i18n_strings` bundled. *(Day 29)*
- **`IonFlow_Pipeline.spec`** — Synced with build_exe.py. *(Day 29)*
- **`installer/ionflow_setup.iss`** — Version bumped to 0.2.0. *(Day 29/30)*

### Build & Release

- PyInstaller build verified (26 MB executable, all data bundled). *(Day 29)*
- Inno Setup installer script updated. *(Day 29)*
- Tag `v0.2.0-rc1` created. *(Day 29)*
- Tag `v0.2.0` created. *(Day 30)*

---

## [0.1.0] — 2026-03-17

Initial release with EIS pipeline, cycling analysis, DRT, PCA, Nyquist/Bode/Ragone
plots, correlation heatmap, interactive GUI, and 2-language i18n (PT/EN).
