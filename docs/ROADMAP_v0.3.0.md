# IonFlow Pipeline — Roadmap v0.3.0

> Revisto em 27 de Abril de 2026 — baseado em análise do código real v0.2.0
> Substitui a versão original (16 Abril 2026) com re-priorização orientada ao utilizador.

---

## 🎯 Visão

Transformar o IonFlow Pipeline numa plataforma de análise eletroquímica **completa e pronta para publicação científica**, com importação nativa de potenciostatos, exportação em formatos de revistas, comparação multi-campanha e IA generativa integrada.

---

## 📅 Cronograma Re-priorizado

| Fase | Feature | Duração | Entregável |
|------|---------|---------|------------|
| **Fase 1** — Quick wins | Relatório LaTeX, pre-commit, PyPI | 1 semana | Débitos técnicos zerados |
| **Fase 2** — Importação direta | Parsers Gamry/BioLogic/Autolab/Zahner | 2 semanas | 4 novos formatos de entrada |
| **Fase 3** — Exportação científica | ZView, LaTeX booktabs, Origin CSV | 1 semana | 3 novos formatos de saída |
| **Fase 4** — Análise comparativa | Aba Comparar + Health Score | 2 semanas | Comparação multi-campanha na GUI |
| **Fase 5** — IA generativa | LLM adapter + narrativa PDF | 2 semanas | Relatórios com texto auto-gerado |
| **Fase 6** — Polish | Painel Settings + auto-update | 1 semana | UX melhorado |
| **Fase 7** — Infraestrutura (opc.) | SQLite + Dashboard Streamlit | 3 semanas | Escalabilidade e acesso web |

**Total estimado: ~9 semanas** (Fases 1–6 obrigatórias; Fase 7 opcional)

---

## 📋 Detalhamento por Fase

### Fase 1 — Quick wins (Semana 1) ✅ EM CURSO

#### 1a: Relatório LaTeX completo
- `report_generator.py` tinha 3 TODOs literais no output `.tex`
- Substituir por conteúdo real: tabelas `booktabs`, resumo executivo, referências BibTeX
- **Status**: implementado nesta sessão
- **Critério de aceitação**: `.tex` gerado compila em `pdflatex` sem erros manuais

#### 1b: Pre-commit hook corrigido
- Black falhava com `PermissionError` no cache → `--no-verify` usado no último commit
- Fix: adicionar `language: system` aos hooks Black e isort (usa venv local)
- **Status**: implementado nesta sessão
- **Critério de aceitação**: `git commit` corre Black/isort sem erros de permissão

#### 1c: Publicar no PyPI
- `pyproject.toml` e `setup.cfg` já estão corretos
- Passos: `python -m build` → `twine upload dist/*`
- Dependência: token PyPI configurado
- **Critério de aceitação**: `pip install ionflow-pipeline` funciona

---

### Fase 2 — Importação Direta (Semanas 2-3)

#### Parser de potenciostatos (`src/parsers/`)
- **Gamry** (`.dta`) — texto estruturado com headers `ZCURVE`
- **BioLogic** (`.mpr`) — binário (usar `galvani` como referência)
- **Autolab** (`.csv`) — detectar headers NOVA/Metrohm
- **Zahner** (`.isc`) — XML/binário proprietário
- Interface comum `PotentiostatParser` com `parse(path) -> pd.DataFrame`
- Auto-detecção por extensão + magic bytes
- Integração na GUI: botão "Importar EIS" aceita estes formatos automaticamente
- **Critério de aceitação**: importar ficheiro real de cada fabricante sem conversão manual

---

### Fase 3 — Exportação Científica (Semana 4)

#### Exportadores (`src/export/`)
- **ZView** (`.z`) — formato texto compatível com ZView/ZPlot
- **LaTeX booktabs** — tabela de parâmetros com incertezas para artigos
- **Origin** — CSV com metadados em comentários `#`
- **MEISP** — formato de texto compatível
- Menu "Exportar como..." na GUI → submenu com todos os formatos
- **Critério de aceitação**: ficheiro exportado abre no software alvo sem modificação

---

### Fase 4 — Análise Comparativa (Semanas 5-6)

#### Aba "Comparar Amostras" na GUI
- Selecionar N amostras da tabela → overlay Nyquist / Bode
- `batch_processor.py` já existe — gap é apenas o frontend
- **Critério de aceitação**: overlay de 5 amostras com legenda automática

#### Timeline de parâmetros
- Gráfico R_ct, C_dl, σ_w por ordem cronológica / ciclo
- Útil para rastrear degradação de eletrodo ao longo do tempo
- **Critério de aceitação**: eixo X = timestamp de aquisição, eixo Y = parâmetro

#### Electrode Health Score (0–100)
- Índice composto: Rs, Rp, KK residual, DRT peaks, cycling retention
- Fórmula normalizada com pesos ajustáveis em `config.py`
- Mostrado como badge colorido na tabela de resultados
- **Critério de aceitação**: score correlaciona com avaliação visual do perito

---

### Fase 5 — IA Generativa (Semanas 7-8)

#### Finalizar LLM adapter (`src/ai/llm_adapter.py`)
- Já existe como módulo experimental; precisа de testes de integração robustos
- Suporte a: **Ollama** (local, offline), **OpenAI** (online, opcional)
- Narrativa automática nas secções do PDF: resumo executivo, interpretação EIS
- Configurável via Settings (modelo, URL, temperatura)
- **Critério de aceitação**: relatório PDF com parágrafo gerado por LLM quando Ollama está ativo

---

### Fase 6 — Polish e UX (Semana 9)

#### Painel Settings na GUI
- `PipelineConfig` editado via JSON à mão actualmente
- Aba "Configurações" com campos para: paths, LLM, fitting bounds, report config
- Salva em `config.json` via `PipelineConfig.save()`
- **Critério de aceitação**: utilizador altera limites de fitting sem editar ficheiros

#### Auto-update funcional (`src/updater.py`)
- Já verifica versão — completar fluxo: download zip → extrai → reinicia GUI
- Especialmente útil no Linux (sem instalador)
- **Critério de aceitação**: notificação de nova versão + botão "Atualizar agora"

---

### Fase 7 — Infraestrutura (Opcional, Semanas 10-12)

#### SQLite backend
- Migrar `FeatureStore` de JSON → SQLite (só vale com dashboard web)
- Para uso local, JSON é suficiente até ~5000 amostras
- Schema: `samples`, `eis_results`, `cycling_results`, `drt_results`, `parameters`

#### Dashboard Streamlit
- Reutilizar funções de visualização existentes
- Páginas: Upload → Pipeline → Resultados → Histórico → IA
- Útil para partilhar com orientador sem instalar Python
- Deploy: `streamlit run dashboard.py` ou Docker

---

## 🏗️ Arquitectura v0.3.0

```
src/
├── parsers/               # NOVO (Fase 2) — Potentiostat importers
│   ├── base.py            # PotentiostatParser interface
│   ├── gamry.py
│   ├── biologic.py
│   ├── autolab.py
│   └── zahner.py
├── export/                # NOVO (Fase 3) — Scientific exporters
│   ├── zview.py
│   ├── latex.py
│   ├── origin.py
│   └── meisp.py
├── db/                    # NOVO (Fase 7, opc.) — SQLite backend
│   ├── models.py
│   ├── migrations.py
│   └── feature_store_v2.py
├── dashboard/             # NOVO (Fase 7, opc.) — Streamlit app
│   ├── app.py
│   └── pages/
└── (35 módulos v0.2.0 mantidos)
```

---

## ✅ Pré-requisitos e Estado Actual

- [x] v0.2.0 released no GitHub (commit `7f3a009`)
- [x] CI a funcionar (Python 3.11/3.12)
- [x] 1782+ testes automatizados
- [x] Cobertura ≥ 95%
- [x] Pre-commit hook corrigido (Fase 1b)
- [x] Relatório LaTeX sem TODOs (Fase 1a)
- [ ] Publicar no PyPI (Fase 1c) — pendente token
- [ ] Tag `v0.3.0-dev` criada

---

## 📊 Métricas de Sucesso

| Métrica | v0.2.0 | v0.3.0 (alvo) |
|---------|--------|----------------|
| Testes | 1782 | 2500+ |
| Cobertura | ≥ 95% | ≥ 95% |
| Formatos importação | CSV, XLSX | + Gamry, BioLogic, Autolab, Zahner |
| Formatos exportação | PDF, Excel, MD | + ZView, LaTeX, Origin, MEISP |
| Health Score | — | 0–100 por eletrodo |
| Relatórios IA | Heurística | + LLM generativo (Ollama) |
| Comparação multi-amostra | — | Overlay na GUI |

---

## ❌ Features Removidas / Re-priorizadas

| Feature original | Decisão | Motivo |
|-----------------|---------|--------|
| SQLite (Rec 9) | Fase 7 opcional | JSON suficiente para uso local |
| mypy strict (Rec 12) | Incremental no CI | Zero valor para utilizador final |
| Testes visuais pytest-mpl (Rec 11) | Backlog | Nenhum utilizador percebe a diferença |
| Plugin system (Rec 13) | Backlog | Elegante mas sem demanda imediata |

---

*Repositório: github.com/Emanuel-963/ubiquitous-jougen*
*Versão actual: v0.2.0 | Próxima: v0.3.0 | Roadmap revisto: 2026-04-27*
