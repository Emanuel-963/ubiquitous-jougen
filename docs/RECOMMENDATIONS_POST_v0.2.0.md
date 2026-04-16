# IonFlow Pipeline — Recomendações Pós-Release v0.2.0

> Gerado em 16 de Abril de 2026, após conclusão do plano de 30 dias.

---

## 🔴 Prioridade ALTA (próximas 1-2 semanas)

### 1. GitHub Release
- Ir ao GitHub → Releases → **Create release** a partir da tag `v0.2.0`
- Anexar:
  - Installer Windows (`.exe`) de `dist/installer/`
  - ZIP portátil de `dist/IonFlow_Pipeline/`
  - Source ZIP (gerado automaticamente pelo GitHub)
- Copiar release notes do `CHANGELOG.md`

### 2. Dia 27 pendente — Docstrings + README
- Adicionar docstrings **NumPy-style** em todas as funções públicas de `src/`
- Atualizar `README.md` com:
  - Badges (CI, version, license)
  - Quick start (3 comandos)
  - Secção "Para Pesquisadores" com fluxo típico
  - Secção "Agente IA" com exemplos de recomendação

### 3. Testes em máquina limpa
- Instalar o executável de `dist/IonFlow_Pipeline/` numa máquina **sem Python**
- Verificar que:
  - A GUI abre correctamente
  - Os dados de `data/knowledge/` e `src/i18n_strings/` estão acessíveis
  - O pipeline EIS executa até ao final
  - Os 3 idiomas (PT/EN/ES) funcionam

---

## 🟡 Prioridade MÉDIA (próximas 2-4 semanas)

### 4. Coverage report
- Executar: `pytest --cov=src --cov-report=html`
- Identificar módulos abaixo de **80%** de cobertura
- Adicionar testes direccionados para os módulos mais críticos

### 5. Publicar no PyPI
- `python -m build && twine upload dist/*`
- Permite instalação com `pip install ionflow-pipeline`
- Útil para investigadores que querem usar o pipeline como biblioteca

### 6. Verificar CI no GitHub
- Confirmar que `.github/workflows/ci.yml` passa nos runners do GitHub
- Testar com Python 3.11 e 3.12 na matrix
- Adicionar badge de CI no README

### 7. Tutoriais v0.2.0
- Actualizar os tutoriais existentes em `tutoriais/` com:
  - Novos features: AI agent, PDF reports, CLI, batch processing
  - Exemplos de `ionflow-cli` para cada comando
  - Demonstração do painel IA na GUI

---

## 🟢 Prioridade BAIXA (roadmap v0.3.0)

### 8. Dashboard web
- Criar interface **Streamlit** ou **Dash** para acesso remoto
- Útil para grupos de investigação com servidor partilhado
- Reutilizar as funções de visualização existentes

### 9. Base de dados SQLite
- Migrar o `FeatureStore` de JSON para **SQLite**
- Melhor escalabilidade com centenas/milhares de amostras
- Suporte a queries SQL para análise histórica

### 10. Importação directa de potenciostatos
- Parsers para formatos nativos:
  - **Gamry** (`.dta`)
  - **BioLogic** (`.mpr`)
  - **Autolab** (`.csv` com headers específicos)
  - **Zahner** (`.isc`)
- Eliminaria a necessidade de pré-formatação manual

### 11. Testes de regressão visual
- Usar `pytest-mpl` para detectar mudanças inesperadas nos gráficos
- Guardar imagens de referência em `tests/baseline_images/`
- Comparar automaticamente no CI

### 12. Type checking completo
- Executar `mypy src/ --strict` e corrigir todos os erros
- Adicionar type hints em funções que ainda não têm
- Garantir segurança de tipos em todo o codebase

### 13. Plugin system
- Permitir que utilizadores adicionem circuitos customizados via plugins
- Interface: `CircuitRegistry.register_plugin(path)` carrega módulo externo
- Documentar como criar um plugin de circuito

### 14. Exportação para formatos científicos
- Exportar resultados em formatos padrão:
  - **ZView** (`.z`) para compatibilidade com software comercial
  - **MEISP** format
  - **LaTeX** tabelas para artigos científicos
  - **Origin** (`.opju`) ou CSV formatado para Origin

### 15. Modo comparativo multi-projecto
- Comparar resultados entre diferentes campanhas experimentais
- Timeline de evolução de parâmetros ao longo de meses
- Dashboard com KPIs do laboratório

---

*Documento criado automaticamente em 16/04/2026*
*Repositório: github.com/Emanuel-963/ubiquitous-jougen*
