# Changelog

All notable changes to the IonFlow Pipeline are documented here.

## [0.4.9] — 2026-05-20  _(Gerador de Dados Sintéticos Expandido — 33 topologias de circuitos)_

### Added

- **FEAT — 22 novas topologias de circuitos estendidos** (`scripts/gen_synthetic_eis.py`):
  O gerador agora cobre **33 classes** (11 base + 22 novas composições). As novas
  topologias cobrem combinações físicas realistas de ZARC, RC, CPE, TLM (De Levie),
  Warburg finito/refletivo, Gerischer e elementos indutivos:
  - EXT-01 `Rs-ZARC-TLM` — ZARC + linha de transmissão porosa
  - EXT-02 `Rs-ZARC-ZARC-Wfinite` — 2 ZARCs + difusão finita
  - EXT-03 `Rs-ZARC-ZARC-Wshort` — 2 ZARCs + Warburg refletivo (coth)
  - EXT-04 `Rs-ZARC-ZARC-Gerischer` — 2 ZARCs + difusão química
  - EXT-05 `Rs-ZARC-ZARC-TLM` — 2 ZARCs + TLM
  - EXT-06 `Rs-RC-ZARC-W` — coating ideal + CPE + Warburg
  - EXT-07 `Rs-ZARC-RC-Wfinite` — CPE arc + RC ideal + difusão finita
  - EXT-08 `Rs-L-ZARC-W` — indutivo + Randles + Warburg
  - EXT-09 `Rs-L-ZARC-Wfinite` — indutivo + difusão finita
  - EXT-10 `Rs-L-ZARC-ZARC` — indutivo + 2 arcos
  - EXT-11 `Rs-ZARC-ZARC-ZARC-W` — 3 ZARCs + Warburg
  - EXT-12 `Rs-ZARC-ZARC-ZARC-Wfinite` — 3 ZARCs + Warburg finito
  - EXT-13 `Rs-ZARC-CPE` — arco CPE + CPE bloqueante
  - EXT-14 `Rs-RC-W` — RC ideal + Warburg
  - EXT-15 `Rs-RC-Wfinite` — RC ideal + Warburg finito
  - EXT-16 `Rs-ZARC-ZARC-CPE` — 2 ZARCs + CPE bloqueante
  - EXT-17 `Rs-RC-ZARC-Wfinite` — coating + CPE + difusão finita
  - EXT-18 `Rs-ZARC-RC-Wshort` — CPE arc + RC + Warburg refletivo
  - EXT-19 `Rs-L-ZARC-ZARC-W` — indutivo + 2 ZARCs + Warburg
  - EXT-20 `Rs-TLM` — De Levie puro
  - EXT-21 `Rs-ZARC-ZARC-ZARC-Gerischer` — 3 ZARCs + Gerischer
  - EXT-22 `Rs-ZARC-TLM-W` — ZARC + TLM + cauda difusiva

- **REFACT — Helpers de blocos primitivos** (`_zarc`, `_rc_par`, `_wfin_b`,
  `_wsho_b`, `_ger_b`, `_tlm_b`) extraídos de funções de modelo isoladas,
  permitindo composição sem repetição numérica.

- **GUI** — Label do diálogo de configuração atualizado de "11 circuitos" → "33 circuitos".

---

## [0.4.8] — 2026-05-20  _(Treino do Classificador ML + Parsers de Potenciostatos)_

### Added

- **FEAT — Botão "Treinar Classificador"** (`gui_app.py` sidebar, row 29, azul):
  lê os arquivos `SYN_*.txt` de `data/raw`, extrai features espectrais,
  treina um `RandomForestClassifier` com validação cruzada estratificada
  (5-fold) e salva o modelo em `data/knowledge/ml_classifier.joblib`.
  Exibe diálogo com número de amostras, classes e acurácia CV ao terminar.

- **FEAT — `CircuitMLSelector.train_from_synthetic(data_dir, model_path)`**
  (`src/ml_circuit_selector.py`): novo classmethod que escaneia `SYN_*.txt`,
  decodifica o rótulo do nome do arquivo (`SYN_{circuit}_{NNN}.txt`),
  extrai os 9 features espectrais via `extract_eis_features_for_ml()`,
  exclui classes com < 5 amostras e ajusta o RandomForest.  Retorna dict
  com `n_samples`, `n_classes`, `classes`, `cv_accuracy`, `model_path`.

- **FEAT — `CircuitMLSelector.save_model(path)` / `load_model(path)`**
  (`src/ml_circuit_selector.py`): persistência do modelo via `joblib`;
  `save_model` serializa modelo + metadados; `load_model` restaura instância
  completa pronta para `predict()` e `explain()`.

- **FEAT — Integração de Parsers de Potenciostatos no `load_eis_file()`**
  (`src/loader.py`): ao receber extensão especializada (`.dta`, `.mpr`,
  `.mpt`, `.ism`, `.isc`, `.idf`, `.dfr`), `load_eis_file()` delega ao
  parser correto de `src.parsers` antes de tentar o fallback CSV genérico.
  Habilita leitura nativa de arquivos Gamry `.dta`, BioLogic `.mpr`/`.mpt`,
  Zahner `.ism`/`.isc` e Solartron `.idf`/`.dfr` sem conversão manual.
  Parsers já existiam em `src/parsers/` (Fase 2 do Roadmap) — esta mudança
  os conecta ao pipeline principal.

## [0.4.7] — 2026-05-20  _(Dados Sintéticos para Treino do ML)_

### Added

- **FEAT — Botão "Gerar Dados Sintéticos"** (`gui_app.py` sidebar, row 27):
  abre diálogo de configuração onde o usuário define quantos arquivos EIS
  por circuito (padrão: 20 × 11 circuitos → 220 arquivos em `data/raw`) e
  quantos arquivos de Ciclagem (padrão: 10 em `data/processed`) deseja gerar.
  A geração roda em segundo plano e popula automaticamente a aba
  **Comparar Amostras** ao terminar.

- **FEAT — Botão "Excluir Dados Sintéticos"** (`gui_app.py` sidebar, row 28):
  conta os arquivos `SYN_*` em `data/raw` e `SYN_CIC_*` em `data/processed`,
  pede confirmação e os remove; atualiza o cache `self.raw_eis` e a checklist
  da aba Comparar Amostras imediatamente.

- **`scripts/gen_synthetic_cycling.py`** (novo): gerador de ciclagem
  galvanostática sintética (GCPL). Simula supercapacitores com parâmetros
  físicos aleatórios (corrente, tensão, capacitância, ESR, taxa de degradação)
  no formato `Time (s);WE(1).Current (A);WE(1).Potential (V);Cycle` esperado
  pelo `cycling_loader.py`. Expõe `generate(n_files, out_dir, seed)` e
  `clean_synthetic(out_dir)`. Também utilizável via CLI:
  `python scripts/gen_synthetic_cycling.py --n 20 --seed 42`.

- **FEAT — Auto-carregamento EIS na inicialização** (incluído nesta release):
  `_autoload_eis_on_startup()` carrega automaticamente os arquivos de `data/raw`
  na abertura do programa (backport do commit `a97c620` introduzido em 0.4.6).

## [0.4.6] — 2026-05-19  _(Rodar Ambos — Timeline, Health Score, Relatório e Auto-Carregamento)_

### Added

- **FEAT — Auto-carregamento de amostras EIS na inicialização** (`gui_app.py`
  `_autoload_eis_on_startup`): ao abrir o programa, os arquivos EIS já
  presentes em `data/raw` (diretório padrão de `PipelineConfig`) são
  carregados automaticamente em segundo plano para `self.raw_eis`, populando
  a aba **Comparar Amostras** sem precisar importar nem rodar o pipeline.

### Fixed

- **BUG-11 — Timeline e Health Score vazios após "Rodar Ambos"** (`gui_app.py`
  `_handle_both_done`): o método extraía `eis_result` e `cic_result` do par
  retornado pelo worker, mas não armazenava nenhum dos dois em
  `self.last_eis_result` / `self.last_cycling_result`. Como a aba Compare e
  o diálogo de relatório dependem dessas referências para detectar dados
  disponíveis, Timeline e Health Score exibiam a mensagem *"Execute o
  Pipeline EIS primeiro"* e as seções EIS, Ciclagem e Correlações do
  diálogo de relatório ficavam desabilitadas *(sem dados)*.
  Correção: duas linhas adicionadas logo após o unpack do resultado:
  `self.last_eis_result = eis_result` e
  `self.last_cycling_result = cic_result`.

- **BUG-12 — Logo não aparecia no relatório gerado** (`gui_app.py`
  `_ask_report_config`, `_generate_report_clicked`): a logo configurada
  em Configurações (`report_logo_path`) nunca era passada para o diálogo
  de exportação nem para `ReportConfig.logo_path`, então a capa do PDF
  ficava sem a imagem. Correção: `_ask_report_config` recebe `logo_path`
  como parâmetro; o diálogo exibe um checkbox "Incluir logo (das
  Configurações)" com miniatura da imagem (marcado por padrão se houver
  logo configurada); ao confirmar, `ReportConfig.logo_path` é preenchido.

## [0.4.5] — 2026-05-19  _(Configurable Report Dialog + Close-Window Fix)_

### Added

- **FEAT-REPORT — Configurable multi-section report dialog** (`gui_app.py`,
  `src/report_generator.py`): "Gerar Relatório" now opens a configuration
  dialog before saving. The researcher can set title / author / institution,
  apply one of four presets (Análise Completa, EIS+DRT, Ciclagem+Fitting,
  IA+KK+EIS), toggle 8 individual sections (EIS, Ciclagem, DRT, Correlações,
  IA, Fitting Report, KK Validation, Referências — disabled automatically
  when no data is available), and choose output format (PDF / Markdown / LaTeX).
- **`ReportConfig`** gains two new boolean flags: `include_fitting_report`
  and `include_kk`. `generate()`, `_generate_pdf()` and `generate_markdown()`
  accept `fitting_report_text` / `kk_text` kwargs and render them as
  sections 5 and 6 (between DRT and Correlations).

### Fixed

- **BUG-10 — Ghost window after close** (`gui_app.py` `_on_close`):
  `_on_close` previously called only `self.destroy()`, which tears down
  the widget tree but does not stop the Tcl/Tk event loop. The
  `_process_queue` callback rescheduled itself every 100 ms, keeping the
  process alive. Fix: store `after()` return IDs (`_after_queue_id`,
  `_after_update_id`), cancel them in `_on_close`, and call `self.quit()`
  after `self.destroy()` to terminate the mainloop.

## [0.4.4] — 2026-05-20  _(Compare Tab Checklist Fix)_

### Fixed

- **BUG-09b — Compare tab checklist blank after import** (`gui_app.py`
  `_refresh_compare_sample_list`): `CTkCheckBox` does not accept a
  `wraplength` keyword — customtkinter raises `ValueError` for unknown
  kwargs, which was silently swallowed by tkinter's event loop and left
  the entire checklist empty.  Removed the invalid `wraplength=180`
  argument; text truncation for long names is already handled by the
  26-char label truncation above the `CTkCheckBox` call.

## [0.4.3] — 2026-05-19  _(Mixed-Format & Compare Tab Fix)_

### Fixed

- **BUG-08 — Pipeline skips non-EIS files gracefully** (`main.py`,
  `main_drt.py`, `src/batch_processor.py`): all three pipelines now filter
  directory entries by extension before calling `load_eis_file()`, and skip
  subdirectories explicitly.  Previously, the presence of any non-EIS file
  (`.xlsx`, `.pdf`, `.png`, JSON, etc.) in the data folder caused silent
  per-file errors that cluttered logs and could prevent valid EIS files from
  contributing results.
  - `EIS_EXTENSIONS` frozenset defined in `src/loader.py` and reused
    everywhere: `.csv .txt .dat .asc .mpt .mpr .dta .idf .z .dfr .ism .isc`
- **BUG-08b — DRT pipeline only read `.txt`** (`main_drt.py`): was filtering
  with `f.lower().endswith(".txt")`; now uses the shared `EIS_EXTENSIONS` set
  so CSV, DAT and vendor formats are included.
- **BUG-08c — BatchProcessor only discovered `.txt` files**
  (`src/batch_processor.py` `_list_files`): same `.txt`-only filter; updated
  to `EIS_EXTENSIONS`.
- **BUG-09 — Compare tab empty after "Importar EIS"** (`gui_app.py`): the
  Import button only copied files to disk; `self.raw_eis` (the source for the
  Compare tab checklist) was never updated until a full pipeline run.  Now,
  after a successful EIS import, a background thread quick-loads the imported
  files into `self.raw_eis` via `_quick_load_eis_dir()` and immediately
  refreshes the Compare tab — no pipeline run required.

## [0.4.2] — 2026-05-18  _(Physical Circuits Edition)_

### Added

- **`fit_verdict` in `fit_template()`** (`src/circuit_fitting.py`): automatic
  three-level rejection verdict (`OK` / `WARNING` / `REJECTED`) based on Orazem 2026
  criteria — `REJECTED` when χ²/ν > 10 OR IC95 > 200% of any parameter OR optimizer
  did not converge; `WARNING` when 5 < χ²/ν ≤ 10 OR IC95 > 100% OR parameter at
  bounds OR structured residuals (autocorr > 0.3).  Now the pipeline **judges**
  fit quality automatically, not just reports it.
- **`MXene-Intercalation` circuit** (13th built-in, `circuit_registry.py`):
  `Rs − (Rsei‖CPEsei) − (Rct‖CPEdl) − Wfinite` — physically grounded model for
  Ti₃C₂Tₓ and Nb₂CTₓ in H₂SO₄/Na₂SO₄.  Three resolved frequency regions:
  surface termination layer (high ω), charge-transfer (mid ω), finite diffusion
  of H⁺/Na⁺ into interlayer spacing (low ω, tanh boundary condition).
- **`De-Levie-TLM` circuit** (14th built-in): full transmission-line model
  `Rs + sqrt(Ri/Ydl)·coth(L·sqrt(Ri·Ydl))` for porous electrodes (MXene films,
  activated carbon, CNTs). Correctly predicts 45° high-ω line transitioning to
  near-vertical capacitive response.  Ref: De Levie (1963) / Tribollet & Orazem §15.
- **`Pseudo-Capacitance-CPE` circuit** (15th built-in):
  `Rs − (Rct‖CPEdl) − (Rads‖Cads)` — adsorption pseudocapacitance model
  for RuO₂, MnO₂, Nb₂O₅, UPD systems.  Distinguishable from Two-Arc-CPE by
  the ideal capacitor Cads (n=1) in the adsorption branch.
- **`fit_verdict` + `fit_verdict_reasons` propagated to `circuit_df`** (`main.py`):
  these fields now appear in the circuit table and are used by the GUI metrological
  summary (previously only chi²/ν threshold was used).
- **GUI metrological summary upgraded** (`gui_app.py`): now prefers `fit_verdict`
  field over raw chi²/ν threshold; shows first reason for WARNING/REJECTED per row;
  criteria legend expanded to show all three verdict levels.

### Changed

- Exception rows in `build_circuit_tables` now carry `fit_verdict=REJECTED` and
  `fit_verdict_reasons=[str(exc)]` for consistent downstream handling.

### Tests

- `tests/test_circuit_registry.py`: updated count assertions 12→15; added
  `MXene-Intercalation`, `De-Levie-TLM`, `Pseudo-Capacitance-CPE` to expected names.

---

## [0.4.1] — 2026-05-18  _(Metrological Rigor Edition)_

### Added

- **Orazem σ-model** (`src/circuit_fitting.py` — `orazem_sigma()`):
  weighted error structure `σ = α·|Zⱼ| + β·|Zᵣ|` with
  α=0.001216, β=0.000333 per Tribollet & Orazem (2026), Eq. 9.
- **Weighted fitting** (`fit_template()`): now accepts `sigma=` keyword;
  residuals are normalised by σ(f) for statistically correct χ²/ν.
- **New output fields** in `fit_template()`: `chi2_over_nu`, `confidence_interval_95`,
  `rel_uncertainty_pct`, `param_significance` (significativo / marginalmente / não-significativo).
- **Orazem noise model in Monte Carlo** (`src/uncertainty.py`): replaced
  flat `noise_pct·|Z|` with `α·|Zⱼ| + β·|Zᵣ|` matching the Tribollet & Orazem structure.
- **Four metrological helpers** (`src/validation.py`):
  - `detect_powerline_noise(freq)` — boolean mask for 50±3 Hz / 100±3 Hz points.
  - `remove_highest_frequency_point(freq, z)` — removes the turn-on-transient artefact.
  - `estimate_critical_frequency(Re, C_inf)` — fc = 1/(2πRe·C∞).
  - `truncate_above_fc(freq, z, fc)` — drops frequencies above fc.
- **Porous-Coating-TLM** (`src/circuit_registry.py`): 12th built-in circuit
  `Rs − (Rcoat‖Cpore) − (Rct‖CPEdl)` based on de Levie (1967) / Gabrielli (1997)
  and Tribollet & Orazem (2026) Eqs. 16–18.
- **CPE-n physical rules** and **zombie-parameter detection** in
  `src/fitting_report.py` (n≈0.5 Warburg warning; IC_95>100% → zombie flag;
  chi2/nu tiered summary ✓/⚠/⛔).
- **`ionflow preprocess` CLI subcommand** (`src/cli.py`): applies powerline filter,
  HF-point removal, and fc-truncation in batch. Supports `--json` output.
- **Powerline check in `ionflow validate`**: now reports count and frequencies of
  50/100 Hz noise points per file.
- **Tutorial 24** (`tutoriais/24_metrologia_orazem_tribollet.txt`): complete walkthrough
  of the Orazem/Tribollet metrological workflow (7 sections, code examples).
- **Reference [EIS-6]** (`tutoriais/08_referencias_bibliograficas.txt`):
  Tribollet & Orazem, Electrochimica Acta 568, 149009 (2026),
  DOI: 10.1016/j.electacta.2026.149009. Added to module reference maps.

### Changed

- Tutorial 02 (Pipeline EIS): added §2.3 "Pré-processamento metrológico" and updated
  circuit table description with `chi2_over_nu`, `confidence_interval_95` fields.
- Tutorial 07 (Interpretação Científica): added metrological pre-check step;
  workflow updated with PASSO 0 (preprocess) and PASSO 1b (χ²/ν check).
- Tutorial 12 (CLI): completely rewritten for v0.4.x CLI syntax (`ionflow` not `ionflow-cli`);
  added `preprocess`, `validate`, `config`, `version` subcommands; JSON integration examples.
- References file: bumped to v0.4.1, added [EIS-6], updated module and tutorial maps.

### Tests

- `tests/test_circuit_registry.py`: updated 7 count assertions from 11→12 for
  the new Porous-Coating-TLM circuit; added `"Porous-Coating-TLM"` to expected names.
- **235 tests passing** (7 files).

---

## [0.4.0] — 2026-05-12  _(Market Edition)_

### Added

- **MKT-03** — Logo 80×80 thumbnail preview in Settings tab before PDF export.
- **MKT-04** — Solartron `.idf` / `.dfr` parser with 13 new tests.
- **MKT-05** — "Export all (.xlsx)" sidebar button: exports up to 8 result sheets
  (EIS, DRT, Cycling, Ranking, PCA…) into a single multi-tab workbook via openpyxl.
- **MKT-07** — SHA-256 fingerprint in PDF report footer
  (`IonFlow Pipeline {version}  |  SHA-256: {fingerprint[:16]}  Page X / N`).
- **MKT-08** — i18n completeness: added `ui.export_all_xlsx`, `ui.license_*`,
  `ui.no_results_to_export`, `ui.exported_n_sheets` keys to all three locale files
  (pt / en / es).
- **LIC-03** — Lab tier key format `IONFLOW-LAB-<SERIAL>-<MAC>` with seat-limit
  constant (`LAB_SEAT_LIMIT = 5`); `is_lab` / `is_oem` properties on `LicenseManager`.
- **LIC-04** — Headless SDK (`src/sdk.py`): `EISAnalyzer`, `DRTAnalyzer`,
  `ReportBuilder` classes — no GUI dependency, suitable for embedding in third-party
  software or REST backends.
- **LIC-05** — Flask validation server (`scripts/license_server.py`): endpoints
  `GET /api/v1/ping`, `POST /api/v1/validate`, `POST /api/v1/validate/batch`.
- **generate_license_key.py** — now accepts `TIER` argument (PRO / LAB / OEM).
- **44 new tests** in `tests/test_license_manager.py` covering all three tiers.

### Fixed

- PDF footer: replaced U+2022 bullet (outside Helvetica Latin-1 range) with `|`.

### Changed

- `validate_key()` now accepts Pro, Lab, and OEM keys (was Pro-only).
- `LicenseManager.activate()` auto-detects tier from key prefix.
- `LicenseManager.status_label()` returns tier-appropriate strings.

---

## [0.3.1] — 2026-05-07

### Fixed

- **BUG-01** — `_handle_both_done` now calls `_refresh_compare_sample_list()` so the
  Compare tab is populated after running the "Both" (EIS + Cycling) pipeline.
- **BUG-02** — Sidebar "Comparar Amostras" button now calls the new `_open_compare_tab()`
  method, which refreshes the checklist before switching tabs.
- **BUG-03** — Streamlit exit code 1 confirmed as false positive (Ctrl+C termination);
  dashboard starts correctly on `http://localhost:8501`.
- **BUG-04** — `build_exe.py` now emits a `warnings.warn` if the build Python is not 3.11,
  guiding users to run `python3.11.cmd build_exe.py`.
- **BUG-05** — `_refresh_compare_sample_list` has an early-exit (using `frozenset` of keys)
  that skips full widget rebuild when the sample set is unchanged.
- **BUG-06** — `_run_compare_clicked` shows an orange `CTkLabel` warning when fewer than
  2 samples are selected, in both the Nyquist and Bode frames.
- **BUG-07** — Added `tutoriais/23_comparar_amostras_gui.txt`: step-by-step guide for
  the Compare tab (prerequisites, 6 steps, edge-case table, FAQ).

### Performance

- **OPT-02** — `src/loader.py` caches `load_eis_file()` results keyed by file path +
  `mtime_ns`. Re-reading an unchanged file is a no-op (returns a DataFrame copy from
  memory). Call `clear_load_cache()` to evict all entries.
- **OPT-03** — `src/comparison/overlay_plots.py` applies adaptive downsampling before
  rendering: if a series has > 200 points, `_downsample()` applies a uniform stride so
  each sample contributes at most `_MAX_PLOT_POINTS = 200` points. Both `plot_nyquist_overlay`
  and `plot_bode_overlay` benefit.
- **OPT-04** — `_refresh_compare_sample_list` caps visible checkboxes at 50
  (`_MAX_VISIBLE = 50`). When additional samples exist, a grey italic label
  `"(+ N mais — use filtro)"` is shown at the bottom of the scroll frame.

### Improved

- **UX-01** — Buttons "✓ Todos" and "✗ Limpar" in the Compare tab were already wired
  correctly to `_compare_select_all` / `_compare_select_none`; confirmed working.
- **UX-02** — Sidebar "Comparar Amostras" button now shows the loaded sample count:
  `"🔄 Comparar Amostras (N)"`. Updates every time the Compare tab is opened.
- **UX-03** — `_refresh_compare_sample_list` preserves the user's checkbox selection
  across pipeline re-runs. Previously-checked samples remain checked; new samples
  default to checked.
- **UX-04** — The "🔄 Comparar" button inside the Compare tab is disabled and shows
  "⏳ Calculando…" while plots are being generated. Restored in a `finally` block so
  the button is never stuck disabled even on error.

- Added `TestCompareTabRegression` (9 tests) to `tests/test_gui_tabs.py` covering
  BUG-01, BUG-02, BUG-05 and BUG-06. All 86 tests pass.

---

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
