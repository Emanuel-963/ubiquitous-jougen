---
title: 'IonFlow Pipeline: An Open-Source Python Tool for Electrochemical Impedance Spectroscopy Analysis, Circuit Fitting, and Battery Cycling Data'
tags:
  - Python
  - electrochemistry
  - impedance spectroscopy
  - equivalent circuit fitting
  - battery cycling
  - DRT
  - open source
authors:
  - name: Emanuel de Souza Silva Oliveira
    orcid: 0009-0007-2573-9628
    affiliation: 1
affiliations:
  - name: Universidade do Vale do Paraíba (UNIVAP), Faculdade de Engenharia, Arquitetura e Urbanismo (FEAU), São José dos Campos, São Paulo, Brasil; Instituto de Aeronáutica e Espaço (IAE), Laboratório de Materiais (AMR), São José dos Campos, São Paulo, Brasil
    index: 1
date: 25 May 2026
bibliography: paper.bib
---

# Summary

Electrochemical impedance spectroscopy (EIS) is a powerful, non-destructive
characterization technique widely used in energy storage research, corrosion
science, and sensor development [@barsoukov2018impedance].
Interpreting EIS data requires fitting measured spectra to equivalent circuit
models and extracting physically meaningful parameters — a process that
historically demanded expensive commercial software.
**IonFlow Pipeline** is an open-source Python desktop application that
provides a complete EIS analysis workflow: multi-format data ingestion,
Kramers-Kronig validity testing, automatic circuit selection and non-linear
least-squares fitting, distribution of relaxation times (DRT) analysis,
principal component analysis (PCA) of extracted features, battery cycling
energy/power calculations, and branded PDF report generation.
The tool targets researchers who need reproducible, scriptable analysis
without vendor lock-in.

# Statement of Need

Commercial EIS analysis packages (RelaxIS, ZView, EC-Lab) cost hundreds to
thousands of euros per seat and produce non-interoperable project files that
hinder data sharing and reproducibility.
Free alternatives are either limited to a single fitting model (EIS Spectrum
Analyser) or require manual scripting without a user interface (impedance.py
[@murbach2020impedancepy]).
IonFlow Pipeline fills the gap by combining a polished desktop GUI with a
fully testable Python library that can be integrated into automated workflows.

Table 1 summarises the feature landscape relative to the most widely used tools
in the community.  A quantitative accuracy benchmark against ZView and
impedance.py on a common reference dataset is planned for the peer-review
cycle and will be reported in an accompanying validation note.

| Feature | IonFlow Pipeline | impedance.py | ZView | EC-Lab |
|---------|:---:|:---:|:---:|:---:|
| Open source | ✓ | ✓ | ✗ | ✗ |
| Desktop GUI | ✓ | ✗ | ✓ | ✓ |
| Kramers-Kronig validation | ✓ | ✓ | ✓ | ✓ |
| DRT analysis | ✓ | ✗ | ✗ | ✗ |
| Monte Carlo uncertainty | ✓ | partial | ✗ | ✗ |
| Batch / CLI pipeline | ✓ | scripting | ✗ | ✗ |
| Multi-vendor parsers | ✓ | partial | ✗ (proprietary) | ✗ (proprietary) |
| Galvanostatic cycling | ✓ | ✗ | ✗ | ✓ |
| PDF report generation | ✓ | ✗ | ✗ | ✓ |
| Rule-based AI agent | ✓ | ✗ | ✗ | ✗ |
| Automated test suite | 2000+ tests | limited | ✗ | ✗ |

: Feature comparison (✓ = supported, ✗ = not supported, partial = limited support) {#tbl:comparison}

Key differentiators:

- **Multi-vendor parser**: reads CSV, Gamry `.dta`, BioLogic `.mpt`/`.mpr`
  (via optional `galvani` dependency), Zahner `.idf`, and generic text formats.
- **Eleven built-in equivalent circuits**: Randles-CPE-W, ZARC-ZARC-W,
  Warburg (finite and short), Gerischer, Three-ZARC, and more, all with
  physically motivated initial parameter estimates and bounds.
- **Automated circuit shortlisting**: an ML-based selector pre-trained on
  synthetic spectra recommends the best-matching topologies before fitting,
  reducing manual trial-and-error.
- **DRT analysis**: regularised Tikhonov inversion of the impedance spectrum
  yields the distribution of relaxation times, providing a model-free view of
  the system's time constants [@wan2015influence; @ciucci2015analysis].
- **Kramers-Kronig validation**: the lin-KK residual method [@schonleber2014method]
  flags non-stationary or non-linear data before fitting.
- **Batch processing and feature store**: automated pipelines over folders of
  files, with fitting results persisted to a SQLite feature store for
  longitudinal tracking.
- **Reproducible reports**: PDF and Markdown reports with custom branding
  (logo, author, institution) generated via `fpdf2`.
- **Full test suite**: >2000 pytest tests covering unit, integration, and
  regression scenarios (100 % pass rate on Python 3.13).

# Implementation

IonFlow Pipeline is structured as a layered Python package under `src/`.
The data flow is:

1. **Parsing** (`src/parsers/`) — vendor-specific readers return a normalised
   three-column DataFrame (frequency, Z′, Z″).
2. **Preprocessing** (`src/preprocessing.py`) — outlier removal, duplicate
   frequency handling, and Kramers-Kronig pre-screening.
3. **Fitting** (`src/circuit_fitting.py`, `src/circuit_registry.py`) —
   `scipy.optimize.curve_fit`-based least-squares fitting with per-circuit
   analytical Jacobians where available.
4. **Diagnostics** (`src/fitting_diagnostics.py`) — residuals, relative errors,
   Nyquist/Bode reconstruction overlays.
5. **DRT** (`src/drt_analysis.py`) — Tikhonov-regularised inversion on a
   logarithmically spaced τ grid.
6. **PCA and ranking** (`src/pca_analysis.py`, `src/ranking.py`) — feature
   matrix assembled from all fitted parameters, projected onto principal
   components, classified by custom thresholds.
7. **GUI** (`gui_app.py`) — CustomTkinter desktop application that exposes
   all the above through a tabbed interface with a live log pane and
   interactive Matplotlib canvases embedded via `FigureCanvasTkAgg`.

The public Python API allows headless use in Jupyter notebooks or CI pipelines:

```python
from main import run_eis_pipeline

result = run_eis_pipeline()        # reads data/raw/, writes data/processed/
df     = result.feature_table      # pandas DataFrame of fitted parameters
```

# Limitations

The current release has several known limitations that users should consider
before relying on results for publication:

**Synthetic-data training for ML components.**
The ML-based circuit shortlisting (`CircuitMLSelector`) and the cycling
performance predictor (`PerformancePredictor`) were pre-trained entirely on
synthetic impedance spectra and cycling curves generated programmatically.
No experimental validation set has been assembled yet.
Users should treat ML-generated suggestions as a starting point for manual
inspection, not as authoritative classifications.
The hierarchical classifier achieves approximately 62 % accuracy on held-out
synthetic spectra; accuracy on real laboratory data is not yet characterised.

**Fitting accuracy not yet benchmarked externally.**
The non-linear least-squares fitting infrastructure has been validated for
correct convergence behaviour (the optimiser converges, returns physically
plausible parameter magnitudes, and residuals decrease monotonically) but has
not been benchmarked against ZView or EC-Lab on a common reference dataset.
Such a comparison is planned and will be reported separately.

**Orazem-Tribollet weighting coefficients.**
The measurement model noise coefficients (α = 0.001216, β = 0.000333)
are taken from @tribollet2023electrochemical and are calibrated for aqueous
electrolyte systems at room temperature.
They should not be applied without re-calibration to non-aqueous electrolytes,
extreme temperatures, or non-planar electrode geometries.

**DRT regularisation parameter λ.**
The Tikhonov regularisation parameter is selected heuristically; no automatic
L-curve or generalised cross-validation procedure is implemented.
DRT output is therefore suitable for qualitative identification of relaxation
time peaks but absolute values of the distribution function γ(τ) should not
be used as quantitative descriptors without expert validation of λ.

**BioLogic `.mpr` parser coverage.**
Parsing of proprietary BioLogic binary files is limited to firmware versions
tested internally.
Files produced by instruments running more recent firmware may parse
incorrectly or raise an exception.

# Acknowledgements

This work received no external funding.

# References
