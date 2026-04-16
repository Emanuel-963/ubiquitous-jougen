# IonFlow Pipeline — Roadmap v0.3.0

> Criado em 16 de Abril de 2026
> Baseado nas recomendações pós-release v0.2.0 (prioridade baixa)

---

## 🎯 Visão

Transformar o IonFlow Pipeline numa **plataforma colaborativa e extensível**
para laboratórios de eletroquímica, com acesso web, base de dados escalável,
importação nativa de potenciostatos e sistema de plugins.

---

## 📅 Cronograma Estimado

| Fase | Recs | Duração | Entregável |
|------|------|---------|------------|
| **Fase 1** — Fundações | 9, 12 | 2 semanas | SQLite + type safety |
| **Fase 2** — Importação | 10 | 2 semanas | Parsers Gamry/BioLogic/Autolab/Zahner |
| **Fase 3** — Qualidade | 11 | 1 semana | Testes de regressão visual |
| **Fase 4** — Extensibilidade | 13, 14 | 2 semanas | Plugin system + exportação científica |
| **Fase 5** — Dashboard | 8, 15 | 3 semanas | Interface web + modo comparativo |

**Total estimado: ~10 semanas**

---

## 📋 Detalhamento por Feature

### Fase 1 — Fundações (Semanas 1-2)

#### Rec 9: Base de dados SQLite
- Migrar `FeatureStore` de JSON → **SQLite**
- Schema: tabelas `samples`, `eis_results`, `cycling_results`, `drt_results`, `parameters`
- Manter compatibilidade retroactiva (importar JSON existentes)
- Suporte a queries SQL para análise histórica
- **Critério de aceitação**: 1000+ amostras sem degradação de performance

#### Rec 12: Type checking completo
- Executar `mypy src/ --strict` e resolver todos os erros
- Adicionar type hints em funções sem anotação
- Configurar `mypy` no CI como check obrigatório (não `continue-on-error`)
- **Critério de aceitação**: `mypy --strict` sem erros

### Fase 2 — Importação Directa (Semanas 3-4)

#### Rec 10: Parsers de potenciostatos
- **Gamry** (`.dta`) — parser de texto estruturado
- **BioLogic** (`.mpr`) — parser binário (usar `galvani` como referência)
- **Autolab** (`.csv`) — detectar headers NOVA/Metrohm
- **Zahner** (`.isc`) — parser XML/binário
- Cada parser implementa interface `PotentiostatParser`
- Auto-detecção de formato baseada em extensão + magic bytes
- **Critério de aceitação**: importar ficheiros reais de cada fabricante

### Fase 3 — Qualidade Visual (Semana 5)

#### Rec 11: Testes de regressão visual
- Instalar `pytest-mpl`
- Gerar imagens baseline para todos os gráficos (Nyquist, Bode, DRT, Ciclagem)
- Guardar em `tests/baseline_images/`
- Adicionar step no CI: `pytest --mpl --mpl-generate-summary=basic`
- Tolerância configurável (default: 2% RMS)
- **Critério de aceitação**: CI falha se gráfico mudar inesperadamente

### Fase 4 — Extensibilidade (Semanas 6-7)

#### Rec 13: Plugin system
- Interface: `CircuitRegistry.register_plugin(path)`
- Formato plugin: módulo Python com função `register(registry)`
- Pasta de plugins: `~/.ionflow/plugins/` ou `plugins/`
- Hot-reload na GUI (botão "Recarregar Plugins")
- Documentação: tutorial `16_criar_plugin_circuito.txt`
- **Critério de aceitação**: plugin externo aparece no dropdown de circuitos

#### Rec 14: Exportação para formatos científicos
- **ZView** (`.z`) — formato texto com headers padrão
- **LaTeX** — tabelas `booktabs` com parâmetros ajustados
- **Origin** — CSV formatado com metadados em comentários
- **MEISP** — formato compatível
- Menu "Exportar como..." na GUI com todos os formatos
- **Critério de aceitação**: ficheiro exportado abre no software alvo

### Fase 5 — Dashboard Web (Semanas 8-10)

#### Rec 8: Dashboard Streamlit
- Reutilizar funções de visualização existentes (`eis_charts`, `cycling_charts`, `drt_charts`)
- Páginas: Upload → Pipeline → Resultados → Histórico → IA
- Autenticação básica (password)
- Deploy: `streamlit run dashboard.py` ou Docker
- **Critério de aceitação**: pipeline completo executável via browser

#### Rec 15: Modo comparativo multi-projecto
- Comparar resultados entre campanhas experimentais
- Timeline de evolução de parâmetros (R_ct, C_dl, σ_w) ao longo do tempo
- Dashboard com KPIs do laboratório (nº amostras, taxa de sucesso, etc.)
- Filtros: data, operador, tipo de célula, eletrólito
- **Critério de aceitação**: gráfico overlay de N amostras seleccionadas

---

## 🏗️ Arquitectura v0.3.0

```
src/
├── db/                    # NOVO — SQLite backend
│   ├── models.py          # SQLAlchemy/dataclass models
│   ├── migrations.py      # Schema versioning
│   └── feature_store_v2.py
├── parsers/               # NOVO — Potentiostat importers
│   ├── base.py            # PotentiostatParser interface
│   ├── gamry.py
│   ├── biologic.py
│   ├── autolab.py
│   └── zahner.py
├── plugins/               # NOVO — Plugin loader
│   ├── loader.py
│   └── examples/
├── export/                # NOVO — Scientific exporters
│   ├── zview.py
│   ├── latex.py
│   ├── origin.py
│   └── meisp.py
├── dashboard/             # NOVO — Streamlit app
│   ├── app.py
│   ├── pages/
│   └── components/
└── (módulos existentes v0.2.0 mantidos)
```

---

## ✅ Pré-requisitos antes de iniciar

- [x] v0.2.0 released no GitHub
- [x] CI a funcionar (Python 3.11/3.12)
- [x] Cobertura ≥ 95% (1819 testes)
- [x] Tutoriais v0.2.0 completos
- [ ] Publicar no PyPI (Rec 5) — **em curso**
- [ ] Tag `v0.3.0-dev` criada

---

## 📊 Métricas de Sucesso

| Métrica | v0.2.0 (actual) | v0.3.0 (alvo) |
|---------|-----------------|----------------|
| Testes | 1819 | 2500+ |
| Cobertura | 95% | ≥ 95% |
| Formatos importação | CSV, XLSX | + Gamry, BioLogic, Autolab, Zahner |
| Formatos exportação | PDF, Excel | + ZView, LaTeX, Origin, MEISP |
| Type errors (mypy) | ~existem | 0 (strict) |
| Interface | GUI + CLI | + Dashboard web |

---

*Repositório: github.com/Emanuel-963/ubiquitous-jougen*
*Versão actual: v0.2.0 | Próxima: v0.3.0*
