# IonFlow Pipeline — Roadmap v0.3.1

> Criado em 7 de Maio de 2026 — pós-release v0.3.0  
> Foco: **correção de bugs, testes de regressão e optimizações de desempenho**.  
> Não há breaking changes. Todas as alterações são backward-compatible.

---

## 🎯 Objectivo da Release

v0.3.1 é uma release de **estabilização e polish** que resolve os bugs identificados durante
os testes manuais de v0.3.0, aumenta a cobertura de testes nas áreas críticas da GUI e do
pipeline, e melhora a responsividade da interface em operações com múltiplas amostras.

---

## 🐛 Bugs Conhecidos (Críticos)

### BUG-01 — Aba "Comparar Amostras" não mostra amostras  
**Módulo:** `gui_app.py` → `_handle_both_done`  
**Severidade:** 🔴 Alta — funcionalidade core inacessível  
**Causa raiz:** O método `_handle_both_done` (pipeline "Ambos": EIS + Ciclagem) atribuía
`self.raw_eis` mas não chamava `_refresh_compare_sample_list()`. O método
`_handle_eis_done` (pipeline EIS isolado) já chamava o refresh, criando comportamento
inconsistente entre os dois fluxos de execução.  
**Status:** ✅ **Corrigido em `gui_app.py`**  
**Validação necessária:** Rodar pipeline "Ambos" → abrir aba Comparar → confirmar
que as checkboxes aparecem.

### BUG-02 — Botão "Comparar Amostras" na sidebar não refresca a lista  
**Módulo:** `gui_app.py` → `btn_compare` command  
**Severidade:** 🟠 Média — lista pode estar desactualizada ao mudar de aba  
**Causa raiz:** O botão da sidebar chamava apenas `self.tabs.set(...)` sem
reconstruir o checklist. Se o utilizador carregava ficheiros e depois clicava
no botão directamente (sem re-correr o pipeline), via a lista vazia.  
**Status:** ✅ **Corrigido** — criado método `_open_compare_tab()` que faz
`_refresh_compare_sample_list()` antes de mudar de aba.

### BUG-03 — Streamlit dashboard falha ao iniciar  
**Módulo:** `dashboard.py`  
**Severidade:** � Falso positivo — dashboard inicia correctamente  
**Causa raiz:** O exit code 1 da sessão anterior foi terminação manual do processo (Ctrl+C),
não um erro de import. Confirmado: `streamlit run dashboard.py` sobe em
`http://localhost:8501` sem erros. Todos os imports de `src.dashboard`,
`src.db.repository` e `src.db.feature_store_v2` passam sem excepção.  
**Status:** ✅ **Falso positivo — encerrado**

### BUG-04 — Build PyInstaller usa Python 3.14 em vez de 3.11  
**Módulo:** `build_exe.py` / `IonFlow_Pipeline.spec`  
**Severidade:** 🟡 Baixa — build funciona mas gera binário com runtime errado  
**Causa raiz:** O `venv` activo aponta para Python 3.14; `pyinstaller` usa o
interpretador do venv em vez do 3.11 especificado como target.  
**Status:** ✅ **Corrigido** — adicionado `warnings.warn` no `__main__` de
`build_exe.py`: se não for Python 3.11, imprime aviso instruindo a usar
`python3.11.cmd build_exe.py`.

---

## 🐛 Bugs Conhecidos (Menores)

### BUG-05 — `_refresh_compare_sample_list` reconstrói todos os widgets a cada chamada  
**Módulo:** `gui_app.py` → `_refresh_compare_sample_list`  
**Severidade:** 🟡 Baixa — imperceptível com < 20 amostras; noticável com > 50  
**Causa raiz:** A implementação destrói todos os filhos de `compare_scroll` e
recria-os do zero. Não há cache nem diffing.  
**Status:** ✅ **Corrigido** — adicionado early-exit:
`if set(raw_eis.keys()) == set(compare_sample_vars.keys()): return`

### BUG-06 — Sem feedback visual quando nenhuma amostra está seleccionada em "Comparar"  
**Módulo:** `gui_app.py` → `_run_compare_clicked`  
**Severidade:** 🟡 Baixa — utilizador pode não perceber porquê o gráfico não aparece  
**Status:** ✅ **Corrigido** — guarda `selected` com check `len < 2` (cobre tanto
0 como 1 amostra). Mostra `CTkLabel` laranja com "⚠ Selecione pelo menos 2 amostras"
directamente nas frames Nyquist e Bode, além de registar no log.

### BUG-07 — Tutoriais em falta para features de v0.3.0 na GUI  
**Módulo:** `tutoriais/`  
**Severidade:** 🟡 Baixa — documentação  
**Status:** ✅ **Corrigido** — criado `tutoriais/23_comparar_amostras_gui.txt` com
passo-a-passo completo da aba Comparar: pré-requisitos, 6 passos de utilização,
comportamentos esperados, dicas e FAQ.

---

## ✅ Testes de Regressão

9 testes adicionados em `tests/test_gui_tabs.py` → classe `TestCompareTabRegression`.
Todos passam (`9 passed in 1.50s`).

| Teste | Cobertura |
|-------|-----------|
| `test_handle_both_done_refreshes_compare_list` | BUG-01 |
| `test_handle_both_done_all_vars_default_true` | BUG-01 (edge case) |
| `test_open_compare_tab_calls_refresh_before_set` | BUG-02 |
| `test_open_compare_tab_empty_raw_eis_still_switches_tab` | BUG-02 (edge case) |
| `test_run_compare_no_selection_shows_warning` | BUG-06 |
| `test_run_compare_one_sample_also_shows_warning` | BUG-06 (1 amostra) |
| `test_refresh_compare_noop_if_unchanged` | BUG-05 |
| `test_refresh_compare_rebuilds_when_sample_added` | BUG-05 (add) |
| `test_refresh_compare_rebuilds_when_sample_removed` | BUG-05 (remove) |

---

## ⚡ Optimizações de Desempenho

### OPT-01 — Early-exit em `_refresh_compare_sample_list`
**Status:** ✅ **Implementado**
```python
# Compara o conjunto completo de chaves para abranger a cap OPT-04:
current_keys = frozenset(self.raw_eis.keys())
if current_keys == getattr(self, "_compare_last_keys", None):
    return
self._compare_last_keys = current_keys
```
**Impacto estimado:** Elimina 100% do trabalho nas chamadas redundantes (ex:
mudança de aba sem novos dados carregados).

### OPT-02 — Cache de `load_eis_file` por mtime de ficheiro
**Status:** ✅ **Implementado em `src/loader.py`**  
**Módulo:** `src/loader.py`  
**Benefício:** Re-execução do pipeline no mesmo ficheiro não relê do disco.  
**Implementação:** Dict `_LOAD_CACHE` de nível módulo com chave `path → (mtime_ns, DataFrame)`.
Se o ficheiro não foi modificado desde a última leitura, devolve uma cópia imediata
(sem I/O). `clear_load_cache()` disponível para testes ou reimport batch.

### OPT-03 — Gráficos de overlay com downsampling adaptativo
**Status:** ✅ **Implementado em `src/comparison/overlay_plots.py`**  
**Módulo:** `src/comparison/overlay_plots.py`  
**Benefício:** Overlay de 10+ amostras com 500+ pontos cada é lento no Matplotlib.  
**Implementação:** Helper `_downsample(df, max_pts=200)` aplicado antes do loop de
plotagem em `plot_nyquist_overlay` e `plot_bode_overlay`. Com stride uniforme,
a forma da curva EIS é preservada. `_MAX_PLOT_POINTS = 200` é configurável.

### OPT-04 — `CTkScrollableFrame` da aba Comparar: cap de 50 checkboxes
**Status:** ✅ **Implementado em `gui_app.py`**  
**Módulo:** `gui_app.py` → `_refresh_compare_sample_list`  
**Benefício:** Com > 100 amostras, a criação de 100 `CTkCheckBox` bloqueava a UI.  
**Implementação:** Cap a 50 checkboxes (`_MAX_VISIBLE = 50`). Se existirem amostras
escondidas, aparece `CTkLabel` cinzento `"(+ N mais — use filtro)"` no fim da lista.

---

## 🔧 Melhorias de UX (sem breaking changes)

### UX-01 — Botões “Seleccionar Todos / Nenhum” na aba Comparar
**Status:** ✅ **Já funcionava** — os botões “✓ Todos” e “✗ Limpar” já estavam ligados
a `_compare_select_all` / `_compare_select_none`, que iteram corretamente
`compare_sample_vars`. Nenhuma alteração necessária.

### UX-02 — Indicar na sidebar quantas amostras estão carregadas
**Status:** ✅ **Implementado em `gui_app.py`**  
`_open_compare_tab` agora chama `btn_compare.configure(text=...)` após o refresh:
```python
n = len(self.raw_eis)
label = f"🔄 {tr('Comparar Amostras')} ({n})" if n else "🔄 " + tr("Comparar Amostras")
self.btn_compare.configure(text=label)
```
Quando não há amostras, o texto mantém-se inalterado.

### UX-03 — Preservar seleção de amostras ao re-rodar o pipeline
**Status:** ✅ **Implementado em `gui_app.py`**  
`_refresh_compare_sample_list` guarda `previously_selected = frozenset(...)` antes de
destruir os widgets. Ao recriar os checkboxes, amostras que já existiam e estavam
seleccionadas mantêm `var.set(True)`; as novas amostras aparecem seleccionadas por
omissão.

### UX-04 — Estado de loading visível nas subtabs de Comparar
**Status:** ✅ **Implementado em `gui_app.py`**  
O botão “🔄 Comparar” dentro da aba foi guardado como `self._btn_run_compare`.
`_run_compare_clicked` desactiva o botão e muda o texto para “⏳ Calculando…” antes
de chamar `_run_compare_inner()`, e restaura-o no bloco `finally` — mesmo que
ocorra uma excepção.

---

## 📋 Checklist de Release v0.3.1

### Pré-release
- [x] BUG-01 validado manualmente (pipeline "Ambos" → aba Comparar)
- [x] BUG-02 validado manualmente (botão sidebar → lista actualizada)
- [x] BUG-03 investigado e resolvido (dashboard.py)
- [x] BUG-04 documentado em README / build_exe.py
- [x] Testes de regressão (BUG-01, BUG-02) passam em CI
- [x] OPT-01 implementado e testado
- [x] OPT-02 implementado e testado (86 testes passam)
- [x] OPT-03 implementado e testado
- [x] OPT-04 implementado e testado

### Release
- [x] `CHANGELOG.md` actualizado com secção `[0.3.1]`
- [ ] Tag `v0.3.1` criada no GitHub
- [ ] Build PyInstaller executado com `python3.11.cmd`
- [ ] Installer Inno Setup recompilado (`ionflow_setup.iss` versão bump)
- [ ] GitHub Release publicada com instalador e ZIP portátil

---

## 📅 Estimativa de Esforço

| Item | Esforço estimado |
|------|-----------------|
| BUG-01 + BUG-02 (já corrigidos) | ✅ 0h |
| BUG-03 — Streamlit debug | 1–2 h |
| BUG-04 — Documentar build 3.11 | 0.5 h |
| BUG-05 + BUG-06 (minor fixes) | 1 h |
| Testes de regressão (4 testes) | 2 h |
| OPT-01 early-exit | 0.5 h |
| OPT-02 cache preprocessing | 2 h |
| OPT-03 downsampling overlay | 1 h |
| UX-01 a UX-04 | 3 h |
| Tutorial 23 | 1 h |
| Build + installer + release | 1 h |
| **Total** | **~13 h** |

---

## 🔗 Dependências e Contexto

- Versão base: `v0.3.0` (tag `v0.3.0`, commit do branch `main`)
- Ficheiros principais afectados: `gui_app.py`, `src/comparison/overlay_plots.py`,
  `src/preprocessing.py`, `dashboard.py`, `tests/test_gui_tabs.py`
- Sem alterações ao `src/db/`, parsers, exporters ou LLM adapter
