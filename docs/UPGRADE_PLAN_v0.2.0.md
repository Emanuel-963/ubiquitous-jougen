# IonFlow Pipeline — Plano de Upgrade v0.2.0

## Roadmap de 1 Mês · Tarefas Diárias · Abril–Maio 2026

---

### Visão geral

Este plano transforma o IonFlow Pipeline de um projeto de IC em uma **ferramenta de
laboratório profissional** com inteligência artificial integrada. Três eixos prioritários:

| Eixo | Objetivo | Semanas |
|------|----------|---------|
| 🧠 **ML Fitting** | Aprendizado sobre amostras, recombinação de circuitos, feedback textual e visual | S1–S2 |
| 🤖 **Agente IA** | Interpretação de variáveis, previsão de melhorias, recomendações para o processo produtivo | S2–S3 |
| 🏗️ **Refatoração Lab-grade** | Arquitetura profissional, config centralizada, CLI, logging, testes, CI/CD | S1–S4 |

---

## Diagnóstico atual (v0.1.0)

| Problema | Severidade | Impacto |
|----------|-----------|---------|
| `PipelineApp` = god-class (4.072 linhas, 60+ métodos) | 🔴 Crítico | Impossível manter/testar |
| `run_eis_pipeline()` = god-function (400 linhas) | 🔴 Crítico | Impossível testar estágios isolados |
| Paths hardcoded (`"data/raw"`, `"outputs/tables"`) em 15+ locais | 🟠 Alto | Não funciona se mover pasta |
| Sem objeto de configuração — 30+ magic numbers espalhados | 🟠 Alto | Impossível personalizar |
| Apenas 3 circuitos fixos, shortlist por regras manuais | 🟠 Alto | Não aprende com dados |
| Labels em português no código de negócio | 🟠 Alto | Quebra i18n |
| Versão definida em 3 locais diferentes | 🟡 Médio | Risco de inconsistência |
| `requirements.txt` duplica `pyproject.toml` | 🟡 Médio | Manutenção dobrada |
| Nenhum logging estruturado — só `print()` | 🟠 Alto | Impossível debugar em produção |
| Sem tipagem nos retornos dos pipelines (`dict` com string keys) | 🟠 Alto | Fácil quebrar |

---

## SEMANA 1 — Fundações: Refatoração + Infraestrutura ML

> **Meta:** Quebrar o monolito, centralizar config, preparar a base de dados para ML.

### Dia 1 (Segunda) — Config centralizada + Single-source version

**Arquivos novos:**
- `src/config.py` — dataclass `PipelineConfig` com TODOS os parâmetros

**Tarefas:**
1. Criar `PipelineConfig` (dataclass) com campos tipados:
   - `data_dir`, `output_dir`, `figures_dir`, `tables_dir`
   - `voltage` (hoje hardcoded 1.0V em `physics_metrics.py`)
   - `n_head` (hoje hardcoded 5)
   - `cap_filter_range` (hoje `(1e-15, 1e-2)`)
   - `kmeans_k`, `min_rows_clustering`, `variance_threshold`
   - `score_weights: dict` (hoje `{Rp:0.35, Rs:-0.25, C:0.25, E:0.15}`)
   - `pca_columns: list`
   - `stability_columns: list`
   - `correlation_columns: list`
   - `drt_lambda`, `drt_n_tau`, `drt_tau_range`
   - `circuit_max_nfev`, `bic_penalty_structured`, `bic_penalty_bound`
   - `scan_rate`, `electrode_mass`, `electrode_area`
   - `language: str`
2. Criar `PipelineConfig.from_json(path)` e `.to_json(path)`
3. Criar `PipelineConfig.default()` com valores atuais
4. Versão única em `src/__init__.py` → importada por `pyproject.toml` via `dynamic`
5. Eliminar `requirements.txt` → só `pyproject.toml`

**Testes:** `tests/test_config.py` — serialização, defaults, validação

---

### Dia 2 (Terça) — Typed result objects + Decomposição do pipeline EIS

**Arquivos novos:**
- `src/models.py` — dataclasses de resultado

**Tarefas:**
1. Criar dataclasses tipadas:
   ```
   EISResult(features_df, circuit_fits_df, ranked_df, cap_energy_df,
             pca_loadings, pca_evr, stability_dict, circuit_summary_df,
             correlation_matrix, config_used)
   CyclingResult(summary_df, excel_paths, figure_paths)
   DRTResult(results_df, peaks_df, summary_df, figure_paths)
   ```
2. Decompor `run_eis_pipeline()` em funções nomeadas:
   - `load_eis_files(config) → list[DataFrame]`
   - `extract_all_features(dfs, config) → DataFrame`
   - `fit_all_circuits(dfs, config) → DataFrame`
   - `classify_and_rank(features_df, config) → DataFrame`
   - `compute_pca(ranked_df, config) → PCAResult`
   - `compute_stability(ranked_df, config) → dict`
   - `generate_eis_outputs(result, config) → None`
3. `run_eis_pipeline(config) → EISResult` vira orquestrador de 10 linhas
4. Atualizar `gui_app.py` para usar `EISResult` em vez de `dict`

**Testes:** `tests/test_pipeline_eis.py` — teste de cada estágio isolado

---

### Dia 3 (Quarta) — Logging estruturado + Sistema de validação de dados

**Arquivos novos:**
- `src/logger.py` — configuração de logging
- `src/validation.py` — validadores de entrada

**Tarefas:**
1. Configurar `logging` com:
   - Handler para arquivo rotativo (`logs/ionflow.log`)
   - Handler para a TextBox da GUI (via queue)
   - Níveis: DEBUG (arquivo), INFO (GUI), WARNING/ERROR (ambos)
   - Formato: `[YYYY-MM-DD HH:MM:SS] [LEVEL] [module] message`
2. Substituir TODOS os `print()` por `logger.info/warning/error`
3. Criar validadores em `src/validation.py`:
   - `validate_eis_dataframe(df)` — colunas, tipos, ranges, NaN
   - `validate_cycling_dataframe(df)` — ciclo, tempo, potencial, corrente
   - `validate_frequency_range(df)` — alerta se < 3 décadas
   - `validate_impedance_quality(df)` — Kramers-Kronig simplificado (residual check)
4. Integrar validadores no pipeline (logo após load)

**Testes:** `tests/test_validation.py`, `tests/test_logger.py`

---

### Dia 4 (Quinta) — Registry de circuitos + Circuitos novos

**Arquivos modificados:** `src/circuit_fitting.py`
**Arquivos novos:** `src/circuit_registry.py`

**Tarefas:**
1. Criar `CircuitRegistry` — pattern Registry para circuitos:
   ```python
   class CircuitRegistry:
       _circuits: dict[str, CircuitTemplate] = {}
       @classmethod
       def register(cls, template: CircuitTemplate): ...
       @classmethod
       def get(cls, name: str) → CircuitTemplate: ...
       @classmethod
       def all(cls) → list[CircuitTemplate]: ...
       @classmethod
       def from_config(cls, config) → list[CircuitTemplate]: ...
   ```
2. Migrar os 3 circuitos existentes para o registry
3. Adicionar **4 novos circuitos**:
   - `Coating-CPE`: Rs − (Rcoat ‖ CPE_coat) − (Rct ‖ CPE_dl) — revestimentos
   - `Warburg-Finite`: Rs − (Rp ‖ CPE) − W_finite — difusão finita
   - `ZARC-ZARC-W`: Rs − ZARC₁ − ZARC₂ − W — interfaces múltiplas complexas
   - `Simple-RC`: Rs − (Rp ‖ C) — caso ideal (baseline)
4. Cada `CircuitTemplate` agora inclui:
   - `description: str` (para feedback textual)
   - `physical_meaning: dict[str, str]` (para cada parâmetro)
   - `typical_systems: list[str]` (ex: "baterias Li-ion", "supercapacitores")

**Testes:** `tests/test_circuit_registry.py`

---

### Dia 5 (Sexta) — Feature store para ML + Base de dados de fittings

**Arquivos novos:**
- `src/feature_store.py` — persistência de features e resultados
- `data/ml/fitting_history.json` — histórico de fittings

**Tarefas:**
1. Criar `FeatureStore` que persiste em JSON/SQLite:
   - Toda execução de pipeline salva: features espectrais, circuito selecionado, parâmetros, BIC, qualidade do fit
   - Schema: `{sample_id, timestamp, spectral_features, circuit_name, params, bic, confidence, user_label}`
2. Criar `FittingHistory`:
   - Armazena todos os fittings já feitos
   - Permite consulta: "para amostras similares, qual circuito foi melhor?"
   - Calcula estatísticas: circuito mais frequente por faixa de features
3. Integrar salvamento automático no pipeline EIS (pós-fitting)
4. Criar método `FittingHistory.similar_samples(features, n=5)` usando distância euclidiana nas features espectrais normalizadas

**Testes:** `tests/test_feature_store.py`

---

### Dia 6 (Sábado) — ML Classifier para shortlist de circuitos

**Arquivos novos:**
- `src/ml_circuit_selector.py`

**Tarefas:**
1. Substituir shortlist heurística por **classificador treinável**:
   - `CircuitMLSelector` com interface:
     ```python
     class CircuitMLSelector:
         def train(self, feature_store: FeatureStore) → None
         def predict(self, features: dict) → list[str]  # ranked circuits
         def confidence(self, features: dict) → dict[str, float]
         def explain(self, features: dict) → str  # textual feedback
     ```
2. Implementar com `RandomForestClassifier` (sklearn):
   - Input: 9 spectral features do `extract_spectral_features()`
   - Output: probabilidade de cada circuito ser o melhor
   - Treinamento: dados do `FeatureStore` (label = circuito com menor BIC)
3. Fallback para heurística quando `len(feature_store) < 30`
4. Gerar **feedback textual**:
   ```
   "Com base em 47 amostras anteriores com perfil espectral similar,
    o modelo Randles-CPE-W tem 78% de probabilidade de ser o melhor.
    Amostras com slope_low=-0.45 e phase_min=-52° tipicamente mostram
    um semicírculo com cauda de Warburg."
   ```

**Testes:** `tests/test_ml_circuit_selector.py` com dados sintéticos

---

### Dia 7 (Domingo) — Testes de integração + CI/CD

**Arquivos novos:**
- `tests/test_integration.py`
- `.github/workflows/ci.yml`
- `tests/fixtures/` com dados sintéticos

**Tarefas:**
1. Criar fixtures de dados sintéticos (EIS, Ciclagem, DRT)
2. Testes de integração end-to-end:
   - `test_eis_pipeline_full()` — load → result, verifica tipos e shapes
   - `test_cycling_pipeline_full()`
   - `test_drt_pipeline_full()`
   - `test_config_round_trip()` — save + load config
3. CI/CD com GitHub Actions:
   - Matrix: Python 3.11, 3.12
   - Steps: lint (flake8) → type-check (mypy) → test (pytest) → coverage
   - Coverage badge no README

---

## SEMANA 2 — ML de Fitting Avançado + Recombinação de Circuitos

> **Meta:** O sistema aprende com cada análise e recomenda/gera circuitos.

### Dia 8 (Segunda) — Recombinação automática de circuitos

**Arquivos novos:**
- `src/circuit_composer.py`

**Tarefas:**
1. Criar `CircuitComposer` — monta circuitos a partir de blocos:
   ```python
   class CircuitBlock:
       name: str          # "ZARC", "Warburg", "Inductor", "CPE"
       impedance: Callable
       n_params: int
       param_names: list[str]
       bounds: list[tuple]

   class CircuitComposer:
       blocks: list[CircuitBlock]
       def compose(self, blocks: list[str], topology: str) → CircuitTemplate
       def enumerate_candidates(self, max_elements=4) → list[CircuitTemplate]
       def auto_select(self, data, max_elements=3) → list[CircuitTemplate]
   ```
2. Blocos fundamentais: R, C, CPE, W, W_finite, L, ZARC
3. Topologias: série, paralelo, série-paralelo
4. `auto_select()` gera candidatos ≤ N elementos, faz fitting rápido (1 seed, 1000 nfev), retorna top-5 por BIC
5. Limitar a 50 combinações por corrida (timeout safety)

**Testes:** `tests/test_circuit_composer.py`

---

### Dia 9 (Terça) — Feedback visual do fitting (diagnóstico rico)

**Arquivos novos:**
- `src/fitting_diagnostics.py`

**Tarefas:**
1. Criar suite de diagnósticos visuais para CADA fit:
   - **Nyquist overlay**: dados + fit + resíduos (3 painéis)
   - **Bode overlay**: |Z| e fase com fit
   - **Residual analysis**: resíduos vs frequência + histograma + QQ-plot
   - **Parameter confidence**: barras de erro dos parâmetros com bounds
   - **Model comparison**: BIC bar chart dos candidatos com confidence
2. Gerar automaticamente e salvar em `outputs/figures/diagnostics/`
3. Exibir na GUI: nova aba "Diagnóstico" no popup de circuitos
4. Incluir **indicadores semáforo**:
   - 🟢 Fit excelente: RSS baixo, resíduos não-estruturados, bounds OK
   - 🟡 Fit aceitável: RSS moderado ou 1-2 bound hits
   - 🔴 Fit problemático: resíduos estruturados, muitos bound hits

**Testes:** `tests/test_fitting_diagnostics.py`

---

### Dia 10 (Quarta) — Feedback textual inteligente do fitting

**Arquivos novos:**
- `src/fitting_report.py`

**Tarefas:**
1. Criar `FittingReportGenerator`:
   ```python
   class FittingReportGenerator:
       def generate(self, fit_result, history, config) → FittingReport

   @dataclass
   class FittingReport:
       summary: str              # Parágrafo resumo
       circuit_justification: str # Por que este circuito
       parameter_interpretation: dict[str, str]  # Significado de cada param
       quality_assessment: str   # Qualidade do fit
       recommendations: list[str] # Sugestões de melhoria
       comparison_with_similar: str # vs amostras anteriores
   ```
2. Templates de interpretação por circuito (do registry):
   - "Rs = 2.66 Ω indica resistência ôhmica baixa, típica de eletrólito H₂SO₄ concentrado"
   - "n = 0.77 sugere superfície com rugosidade moderada (ideal=1.0, Warburg=0.5)"
   - "Sigma = 45.7 Ω·s^(-1/2) indica contribuição de difusão semi-infinita significativa"
3. Comparação automática com `FeatureStore`:
   - "Esta amostra tem Rs 35% menor que a média das amostras em H₂SO₄"
4. Integrar na GUI: painel de texto abaixo do gráfico de fitting

**Testes:** `tests/test_fitting_report.py`

---

### Dia 11 (Quinta) — Incerteza de parâmetros + Monte Carlo

**Arquivos novos:**
- `src/uncertainty.py`

**Tarefas:**
1. Implementar **Monte Carlo Error Propagation**:
   - Adicionar ruído gaussiano ao espectro (1-5% de |Z|)
   - Re-fitar N vezes (N=100 default, configurável)
   - Calcular média ± std de cada parâmetro
   - Gerar distribuição posterior dos parâmetros
2. Implementar **Bootstrap de resíduos**:
   - Resample dos resíduos do fit
   - Re-fitar com resíduos permutados
   - Intervalo de confiança 95%
3. Adicionar colunas de incerteza ao `circuit_fits.csv`:
   - `Rs_fit ± Rs_std`, `Rp_fit ± Rp_std`, etc.
4. Visualização: elipses de confiança no espaço de parâmetros

**Testes:** `tests/test_uncertainty.py`

---

### Dia 12 (Sexta) — Kramers-Kronig validation

**Arquivos novos:**
- `src/kramers_kronig.py`

**Tarefas:**
1. Implementar teste de Kramers-Kronig linear (método Boukamp):
   - Ajustar série de (R‖C) em cascata (Voigt elements)
   - Calcular resíduos ΔRe e ΔIm
   - Classificar: |residual| < 1% → excelente, < 5% → aceitável, > 5% → suspeito
2. Gerar gráfico de resíduos KK vs frequência
3. Integrar como **pré-validação** no pipeline EIS:
   - Antes do fitting, rodar KK
   - Se falhar, alertar no log e no relatório
   - Flag na tabela: `KK_valid: True/False`
4. Nova aba na GUI: "Validação KK" com gráfico e diagnóstico

**Testes:** `tests/test_kramers_kronig.py`

---

### Dia 13 (Sábado) — Refatoração da GUI: Parte 1 — MVC split

**Arquivos novos:**
- `src/gui/controller.py`
- `src/gui/models.py` (view-models)
- `src/gui/main_window.py`
- `src/gui/__init__.py`

**Tarefas:**
1. Criar package `src/gui/` com separação MVC:
   - **Model** (`src/gui/models.py`): `AppState` dataclass com todos os DataFrames, config, status
   - **Controller** (`src/gui/controller.py`): `PipelineController` com lógica de orquestração
   - **View** (`src/gui/main_window.py`): `MainWindow` só com layout e bindings
2. Mover da classe `PipelineApp`:
   - Dados (DataFrames, results) → `AppState`
   - Lógica (run pipeline, process results) → `PipelineController`
   - UI (widgets, layout) → `MainWindow`
3. `PipelineController` emite eventos que a `MainWindow` escuta
4. Manter compatibilidade: `gui_app.py` importa e chama `MainWindow`

---

### Dia 14 (Domingo) — Refatoração da GUI: Parte 2 — Tab modules

**Arquivos novos:**
- `src/gui/tabs/` — um módulo por grupo de abas
- `src/gui/tabs/eis_charts.py`
- `src/gui/tabs/cycling_charts.py`
- `src/gui/tabs/drt_charts.py`
- `src/gui/tabs/advanced_charts.py`
- `src/gui/tabs/tables.py`
- `src/gui/widgets.py` — widgets reutilizáveis

**Tarefas:**
1. Extrair cada grupo de abas do popup interativo para módulo próprio
2. Criar `BaseChartTab(CTkFrame)` com:
   - Canvas matplotlib compartilhado
   - Toolbar de navegação
   - Padrão mplcursors (eliminar duplicação ×20)
   - Export PNG/SVG com um clique
3. Extrair `StyledOptionMenu`, `FilterableTable`, `LogRedirector` para `widgets.py`
4. `gui_app.py` reduz de 4.072 para ~500 linhas (bootstrap + routing)

---

## SEMANA 3 — Agente IA de Interpretação Eletroquímica

> **Meta:** O software interpreta os resultados e recomenda ações sobre o processo.

### Dia 15 (Segunda) — Knowledge base eletroquímica

**Arquivos novos:**
- `src/ai/knowledge_base.py`
- `src/ai/__init__.py`
- `data/knowledge/electrochemistry_rules.json`

**Tarefas:**
1. Criar base de conhecimento estruturada:
   ```python
   @dataclass
   class ElectrochemicalRule:
       condition: str        # ex: "Rs > 10 Ω"
       interpretation: str   # ex: "Resistência ôhmica elevada"
       possible_causes: list[str]
       recommendations: list[str]
       severity: str         # "info", "warning", "critical"
       references: list[str] # artigos
   ```
2. Cadastrar **50+ regras** cobrindo:
   - Rs (alto/baixo) → causas e soluções
   - Rp (alto/baixo) → transferência de carga
   - n do CPE (>0.9, 0.7-0.9, 0.5-0.7, <0.5) → tipo de interface
   - Sigma (Warburg) → difusão
   - Retenção cíclica (>95%, 80-95%, <80%) → estabilidade
   - DRT picos (posição, largura, amplitude) → processos
   - Correlações anômalas (Rs alto + C alto → contato ruim)
3. Regras parametrizáveis pelo `PipelineConfig` (thresholds ajustáveis)

---

### Dia 16 (Terça) — Motor de inferência rule-based

**Arquivos novos:**
- `src/ai/inference_engine.py`

**Tarefas:**
1. Criar `InferenceEngine`:
   ```python
   class InferenceEngine:
       def analyze(self, eis_result, cycling_result=None, drt_result=None) → AnalysisReport

   @dataclass
   class AnalysisReport:
       findings: list[Finding]        # O que foi encontrado
       anomalies: list[Anomaly]       # O que é incomum
       recommendations: list[Recommendation]  # O que fazer
       quality_score: float           # 0-100
       summary: str                   # Parágrafo executivo
   ```
2. `Finding`: observação factual ("Rs = 2.66 Ω, classificado como baixo")
3. `Anomaly`: desvio do esperado ("Rp negativo indica fitting não convergiu")
4. `Recommendation`: ação sugerida com prioridade:
   - "Reduzir resistência de contato: polir eletrodo ou usar cola de prata"
   - "Repetir medição com faixa de frequência maior (10 mHz – 1 MHz)"
   - "O eletrólito Na₂SO₄ mostra Rs 4.5× maior que H₂SO₄ — considerar trocar"
5. Cross-pipeline: se EIS + Ciclagem disponíveis:
   - "Rs baixo mas potência baixa → gargalo pode ser difusão (verificar DRT)"
   - "Retenção caiu 20% mas Rp estável → degradação mecânica, não eletroquímica"

---

### Dia 17 (Quarta) — Predição de performance

**Arquivos novos:**
- `src/ai/performance_predictor.py`

**Tarefas:**
1. Criar `PerformancePredictor`:
   ```python
   class PerformancePredictor:
       def predict_cycling_from_eis(self, eis_features) → CyclingPrediction
       def predict_degradation(self, eis_before, eis_after) → DegradationPrediction
       def recommend_improvements(self, current_results) → list[Improvement]
   ```
2. `predict_cycling_from_eis()`:
   - Modelo: regressão (Ridge/RandomForest) treinado no FeatureStore
   - Input: Rs, Rp, C, n, Sigma, tau, Dispersion
   - Output: energia estimada, potência estimada, retenção estimada
   - "Com base em suas 15 amostras anteriores, esta impedância sugere energia ~12 μJ e retenção ~85%"
3. `predict_degradation()`:
   - Compara EIS antes/depois de ciclagem
   - Calcula ΔRs, ΔRp, Δn, ΔC
   - Classifica mecanismo: "crescimento de filme", "perda de material ativo", "degradação de contato"
4. `recommend_improvements()`:
   - Baseado nos gargalos identificados:
   - Rs alto → "Melhorar contato elétrico", "Usar eletrólito mais condutivo"
   - n baixo → "Otimizar morfologia da superfície", "Aumentar área ativa"
   - Retenção baixa → "Reduzir janela de potencial", "Adicionar agente estabilizante"

**Testes:** `tests/test_performance_predictor.py`

---

### Dia 18 (Quinta) — Recomendações para o processo produtivo

**Arquivos novos:**
- `src/ai/process_advisor.py`

**Tarefas:**
1. Criar `ProcessAdvisor`:
   ```python
   class ProcessAdvisor:
       def analyze_material_system(self, all_results, metadata) → ProcessReport

   @dataclass
   class ProcessReport:
       material_assessment: str        # Avaliação geral do material
       best_conditions: dict           # Melhor eletrólito, tratamento, etc.
       bottleneck_analysis: str        # Principal limitante
       production_recommendations: list[ProductionRec]
       comparison_table: DataFrame     # Comparativo entre condições
       next_experiments: list[str]     # Sugestões de próximos experimentos
   ```
2. `production_recommendations`:
   - Análise por metadata (eletrólito, tratamento, substrato):
     - "H₂SO₄ proporciona Rs 78% menor → priorizar como eletrólito"
     - "Tratamento térmico (GCD) reduz n_peaks no DRT de 3→2, sugerindo simplificação da interface"
     - "Amostra pós-álcool (NF) tem melhor score — tratamento com álcool melhora interface"
   - Análise de tendências entre réplicas
   - Detecção de outliers (amostra fora do padrão → problema na fabricação?)
3. `next_experiments`:
   - "Testar concentração de H₂SO₄ entre 0.5M e 2M para otimizar Rs"
   - "Medir EIS em mais frequências (< 10 mHz) para resolver difusão lenta"
   - "Repetir ciclagem com 5000 ciclos para avaliar estabilidade longa"

**Testes:** `tests/test_process_advisor.py`

---

### Dia 19 (Sexta) — GUI do Agente IA

**Arquivos novos:**
- `src/gui/tabs/ai_panel.py`

**Tarefas:**
1. Nova aba principal na GUI: **"🤖 Análise IA"**
2. Layout:
   ```
   ┌─────────────────────────────────────────────────────┐
   │  🤖 Análise Inteligente                             │
   ├─────────────┬───────────────────────────────────────┤
   │             │  📊 Resumo Executivo                  │
   │  [Executar  │  ─────────────────────────            │
   │   Análise]  │  "A amostra Nb₂/H₂SO₄ apresenta..." │
   │             │                                       │
   │  Escopo:    │  ⚠️ Anomalias (2)                    │
   │  ☑ EIS      │  • Rs do Na₂SO₄ é 4.5× maior...     │
   │  ☑ Ciclagem │  • Rp convergiu para limite...        │
   │  ☑ DRT      │                                       │
   │             │  💡 Recomendações (5)                  │
   │  Detalhe:   │  1. [ALTA] Reduzir Rs: polir...      │
   │  ○ Resumo   │  2. [MÉDIA] Expandir faixa de f...   │
   │  ● Completo │  3. [MÉDIA] Repetir com 5000 ciclos  │
   │             │                                       │
   │  [Exportar  │  🔮 Previsões                         │
   │   PDF]      │  • Energia estimada: 12±3 μJ         │
   │             │  • Retenção estimada: 85±8%           │
   └─────────────┴───────────────────────────────────────┘
   ```
3. Botão "Exportar PDF" gera relatório completo com todos os findings
4. Tooltip em cada recomendação com a justificativa e referência

---

### Dia 20 (Sábado) — LLM integration layer (opcional, modular)

**Arquivos novos:**
- `src/ai/llm_adapter.py`

**Tarefas:**
1. Criar interface abstrata para LLM (opcional):
   ```python
   class LLMAdapter(ABC):
       @abstractmethod
       def interpret(self, context: str, question: str) → str

   class OpenAIAdapter(LLMAdapter): ...   # API key do usuário
   class OllamaAdapter(LLMAdapter): ...   # Local, sem custo
   class NullAdapter(LLMAdapter): ...     # Fallback: usa rule-based
   ```
2. Se configurado, o LLM enriquece o relatório:
   - Resumo executivo mais natural
   - Comparação com literatura
   - Sugestões mais criativas
3. Se NÃO configurado, usa `NullAdapter` → sistema funciona 100% offline
4. Configuração na GUI: Settings → IA → Provedor (Nenhum / OpenAI / Ollama)

**Testes:** `tests/test_llm_adapter.py` com mock

---

### Dia 21 (Domingo) — Testes + estabilização Semana 3

**Tarefas:**
1. Rodar suite completa de testes (meta: >90% coverage nos módulos novos)
2. Testes de integração do agente IA com dados reais
3. Verificar que pipeline funciona SEM IA configurada (graceful degradation)
4. Code review: eliminar TODOs, normalizar docstrings
5. Commit e tag: `v0.2.0-alpha`

---

## SEMANA 4 — Polimento Lab-grade + Release

> **Meta:** Produto final de qualidade profissional para uso rotineiro em laboratório.

### Dia 22 (Segunda) — CLI profissional

**Arquivos novos:**
- `src/cli.py`

**Tarefas:**
1. CLI com `argparse`:
   ```
   ionflow eis --data-dir data/raw --config config.json --output outputs/
   ionflow cycling --data-dir data/processed --scan-rate 0.1
   ionflow drt --data-dir data/raw --lambda 1e-3
   ionflow analyze --all --ai --export-pdf report.pdf
   ionflow config --init  # gera config.json com defaults
   ionflow validate --data-dir data/raw  # roda KK + validação
   ```
2. Cada comando imprime progresso com barra (tqdm)
3. Saída JSON opcional (--json) para integração com outros tools
4. Return codes padronizados (0=ok, 1=error, 2=warning)

**Testes:** `tests/test_cli.py`

---

### Dia 23 (Terça) — Geração de relatório PDF

**Arquivos novos:**
- `src/report_generator.py`

**Dependência nova:** `reportlab` ou `fpdf2`

**Tarefas:**
1. `ReportGenerator`:
   - Capa com logo, título, data, autor (configurável)
   - Seção EIS: tabela de ranking, melhor circuito, PCA 2D (imagem)
   - Seção Ciclagem: Ragone, energia/potência por ciclo (imagem)
   - Seção DRT: espectros, tabela de picos (imagem)
   - Seção Correlações: heatmap (imagem), top-5 correlações
   - Seção IA: resumo executivo, recomendações, previsões
   - Referências (do tutorial 08)
2. Botão na GUI: "📄 Gerar Relatório PDF"
3. Template configurável (cores, fonte, layout)

**Testes:** `tests/test_report_generator.py`

---

### Dia 24 (Quarta) — Batch processing + Paralelismo

**Arquivos novos:**
- `src/batch_processor.py`

**Tarefas:**
1. `BatchProcessor`:
   - Processa múltiplas pastas de dados em sequência ou paralelo
   - `concurrent.futures.ProcessPoolExecutor` para fitting (CPU-bound)
   - Progresso por amostra na GUI
2. Fitting paralelo dentro de uma corrida:
   - Cada circuito candidato é fitado em processo separado
   - Speed-up: ~3× para 7 circuitos
3. Cancellation support: botão "Parar" na GUI mata os workers
4. Memory guard: limitar workers a `min(cpu_count-1, 4)`

**Testes:** `tests/test_batch_processor.py`

---

### Dia 25 (Quinta) — i18n completo + Espanhol

**Arquivos modificados:** `src/i18n.py`

**Tarefas:**
1. Extrair TODAS as strings hardcoded do código de negócio para i18n:
   - Labels do K-Means ("Interface eficiente", "Genérica estável")
   - Nomes de colunas em CSVs
   - Mensagens de log
   - Textos do agente IA
2. Adicionar idioma **Espanhol** (es)
3. Estrutura: `src/i18n_strings/{pt.json, en.json, es.json}`
4. Cada JSON com seções: `ui`, `pipeline`, `ai`, `reports`, `columns`
5. GUI: seletor de idioma no Settings com preview em tempo real

---

### Dia 26 (Sexta) — Keyboard shortcuts + Acessibilidade

**Tarefas:**
1. Atalhos de teclado:
   - `Ctrl+1/2/3`: Pipeline EIS / Ciclagem / DRT
   - `Ctrl+Shift+A`: Análise IA
   - `Ctrl+E`: Exportar PDF
   - `Ctrl+G`: Abrir gráficos interativos
   - `Ctrl+S`: Salvar configuração
   - `F5`: Re-executar último pipeline
   - `Escape`: Cancelar pipeline em execução
2. Status bar na GUI com:
   - Pipeline em execução / idle
   - Número de amostras carregadas
   - Última análise IA
   - Versão do software
3. Tooltips em TODOS os botões e campos
4. Fonte configurável (acessibilidade: tamanho 12-20)

---

### Dia 27 (Sábado) — Documentação completa + Docstrings

**Tarefas:**
1. Docstrings NumPy-style em TODAS as funções públicas:
   ```python
   def fit_circuit(template, freq, z_data, config=None):
       """Fit an equivalent circuit model to impedance data.

       Parameters
       ----------
       template : CircuitTemplate
           Circuit model to fit.
       freq : np.ndarray
           Frequency array in Hz (descending order).
       z_data : np.ndarray
           Complex impedance array (Z' + jZ'').
       config : PipelineConfig, optional
           Configuration. Uses defaults if None.

       Returns
       -------
       FitResult
           Fitted parameters, diagnostics, and quality metrics.
       """
   ```
2. Atualizar README.md com:
   - Badges (CI, coverage, version, license)
   - Screenshots da GUI
   - Quick start (3 comandos)
   - Seção "Para pesquisadores" com fluxo típico
   - Seção "Agente IA" com exemplos de recomendação
3. Atualizar tutoriais com novos features

---

### Dia 28 (Domingo) — Testes finais + Bug bash

**Tarefas:**
1. Meta de cobertura: **85%** em módulos novos, **70%** geral
2. Bug bash: testar manualmente todos os fluxos:
   - Pipeline EIS com dados vazios, 1 arquivo, 100 arquivos
   - Pipeline Ciclagem com formatos diferentes
   - DRT com diferentes λ
   - Agente IA com/sem ciclagem disponível
   - CLI todos os comandos
   - PDF gerado corretamente
3. Performance: pipeline EIS com 50 arquivos < 60 segundos
4. Memory: GUI não excede 500 MB com 50 amostras carregadas

---

### Dia 29 (Segunda) — Build + Installer + Release candidate

**Tarefas:**
1. Atualizar `pyproject.toml` com novas dependências:
   - `fpdf2>=2.7` (PDF)
   - `tqdm>=4.65` (CLI progress)
2. Atualizar `IonFlow_Pipeline.spec` para PyInstaller:
   - Incluir `data/knowledge/`, `src/i18n_strings/`, `themes/`
3. Build PyInstaller → testar executável
4. Build Inno Setup → testar installer
5. Testar em máquina limpa (sem Python instalado)
6. Tag: `v0.2.0-rc1`

---

### Dia 30 (Terça) — Release v0.2.0

**Tarefas:**
1. CHANGELOG.md com todas as mudanças
2. Atualizar versão para `0.2.0` (single source)
3. Tag `v0.2.0` no git
4. GitHub Release com:
   - Installer Windows (.exe)
   - ZIP portátil
   - Source zip
   - Release notes detalhadas
5. Atualizar ONE_PAGER.md e PRESENTATION.md

---

## Estrutura final esperada (v0.2.0)

```
eis_analytics/
├── src/
│   ├── __init__.py              # Versão única
│   ├── config.py                # 🆕 PipelineConfig centralizado
│   ├── models.py                # 🆕 Dataclasses tipadas
│   ├── validation.py            # 🆕 Validação de entrada + KK
│   ├── kramers_kronig.py        # 🆕 Teste K-K
│   ├── logger.py                # 🆕 Logging estruturado
│   ├── loader.py
│   ├── preprocessing.py
│   ├── physics_metrics.py
│   ├── cpe_fit.py
│   ├── circuit_registry.py      # 🆕 7 circuitos extensíveis
│   ├── circuit_fitting.py       # ♻️ Refatorado
│   ├── circuit_composer.py      # 🆕 Recombinação automática
│   ├── fitting_diagnostics.py   # 🆕 Diagnóstico visual rico
│   ├── fitting_report.py        # 🆕 Feedback textual
│   ├── uncertainty.py           # 🆕 Monte Carlo + Bootstrap
│   ├── feature_store.py         # 🆕 Histórico ML
│   ├── ml_circuit_selector.py   # 🆕 ML shortlist
│   ├── ranking.py               # ♻️ Pesos configuráveis
│   ├── stability.py
│   ├── pca_analysis.py
│   ├── drt_analysis.py
│   ├── drt_visualization.py
│   ├── cycling_calculator.py
│   ├── cycling_loader.py
│   ├── cycling_plotter.py
│   ├── eis_plots.py
│   ├── visualization.py
│   ├── i18n.py                  # ♻️ 3 idiomas
│   ├── metadata.py
│   ├── updater.py
│   ├── batch_processor.py       # 🆕 Paralelo
│   ├── report_generator.py      # 🆕 PDF
│   ├── cli.py                   # 🆕 CLI profissional
│   ├── ai/                      # 🆕 Agente IA
│   │   ├── __init__.py
│   │   ├── knowledge_base.py    # 50+ regras eletroquímicas
│   │   ├── inference_engine.py  # Motor de inferência
│   │   ├── performance_predictor.py  # Previsão de performance
│   │   ├── process_advisor.py   # Recomendações produtivas
│   │   └── llm_adapter.py       # LLM opcional
│   └── gui/                     # 🆕 GUI modular
│       ├── __init__.py
│       ├── controller.py
│       ├── models.py
│       ├── main_window.py
│       ├── widgets.py
│       └── tabs/
│           ├── eis_charts.py
│           ├── cycling_charts.py
│           ├── drt_charts.py
│           ├── advanced_charts.py
│           ├── tables.py
│           └── ai_panel.py
├── data/
│   ├── knowledge/               # 🆕
│   │   └── electrochemistry_rules.json
│   └── ml/                      # 🆕
│       └── fitting_history.json
├── tests/                       # ♻️ Expandido
│   ├── fixtures/                # 🆕 Dados sintéticos
│   ├── test_config.py           # 🆕
│   ├── test_validation.py       # 🆕
│   ├── test_circuit_registry.py # 🆕
│   ├── test_circuit_composer.py # 🆕
│   ├── test_ml_circuit_selector.py # 🆕
│   ├── test_fitting_diagnostics.py # 🆕
│   ├── test_uncertainty.py      # 🆕
│   ├── test_kramers_kronig.py   # 🆕
│   ├── test_inference_engine.py # 🆕
│   ├── test_performance_predictor.py # 🆕
│   ├── test_process_advisor.py  # 🆕
│   ├── test_integration.py      # 🆕
│   ├── test_cli.py              # 🆕
│   ├── test_report_generator.py # 🆕
│   └── ... (19 testes existentes)
├── .github/workflows/ci.yml     # 🆕
└── CHANGELOG.md                 # 🆕
```

---

## Métricas de sucesso

| Métrica | v0.1.0 | v0.2.0 (meta) |
|---------|--------|---------------|
| Linhas `gui_app.py` | 4.072 | < 500 |
| Linhas `main.py` | 444 | < 50 |
| Circuitos disponíveis | 3 | 7+ (extensível) |
| Magic numbers no código | 30+ | 0 (todos em Config) |
| Cobertura de testes | ~70% | > 85% |
| Idiomas | 2 (pt, en) | 3 (+ es) |
| Módulos Python | 18 | 35+ |
| Feedback textual do fitting | ❌ | ✅ Detalhado |
| Agente IA | ❌ | ✅ Rule-based + ML + LLM |
| Relatório PDF | ❌ | ✅ Automático |
| CLI | ❌ | ✅ Completa |
| Logging | print() | ✅ Estruturado |
| Paralelismo | ❌ | ✅ multiprocessing |
| Kramers-Kronig | ❌ | ✅ Validação automática |
| Incerteza de parâmetros | ❌ | ✅ Monte Carlo |

---

## Dependências novas (v0.2.0)

| Pacote | Propósito | Tamanho |
|--------|----------|---------|
| `fpdf2>=2.7` | Geração de PDF | ~2 MB |
| `tqdm>=4.65` | Barra de progresso CLI | ~300 KB |
| `requests>=2.28` | Updater + LLM API | ~500 KB |
| (opcional) `openai>=1.0` | Integração LLM | ~1 MB |

Total adicional estimado: ~4 MB (sem LLM) ou ~5 MB (com LLM)

---

*Plano criado em Abril 2026 para o IonFlow Pipeline v0.2.0*
*Repositório: github.com/Emanuel-963/ubiquitous-jougen*
