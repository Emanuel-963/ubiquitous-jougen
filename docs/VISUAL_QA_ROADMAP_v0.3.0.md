# IonFlow Pipeline v0.3.0 — Roteiro de Testes Visuais

> **Data:** 28 de Abril de 2026
> **Escopo:** GUI (`gui_app.py` — CustomTkinter) + Dashboard (`dashboard.py` — Streamlit)
> **Objetivo:** Identificar erros visuais, de layout, de navegação e de dados antes de qualquer release.

---

## Como usar este roteiro

1. Abra a GUI e o Dashboard **em paralelo** em telas distintas (ou janelas separadas).
2. Siga cada checklist **na ordem apresentada** — muitos testes dependem do estado anterior.
3. Marque `[x]` em cada item ao passar, `[!]` se encontrar defeito, e anote abaixo o defeito.
4. Ao final, registre bugs encontrados na seção **Log de Defeitos**.

---

## Pré-requisitos de ambiente

```powershell
# Ativar venv
.\venv\Scripts\Activate.ps1

# Garantir que não há dados antigos que possam mascarar erros
# (NÃO limpe data/raw — use arquivos reais de EIS/ciclagem)

# Lançar GUI
python gui_app.py

# Lançar Dashboard (outro terminal)
python -m streamlit run dashboard.py
```

---

## BLOCO 1 — GUI: Inicialização e Layout Geral

### T-G-001 — Janela principal

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 1.1 | Janela abre sem traceback no terminal | | |
| 1.2 | Título da barra: `IonFlow Pipeline` | | |
| 1.3 | Tamanho inicial ≈ 1400×900 px (não menor que 1200×800) | | |
| 1.4 | Sidebar visível à esquerda com todos os botões sem sobreposição | | |
| 1.5 | Área de tabs (direita) ocupa o espaço restante sem cortar | | |
| 1.6 | Sem scroll horizontal na janela principal | | |
| 1.7 | Tema escuro carregado (fundo escuro, texto claro) | | |
| 1.8 | Logotipo/ícone na barra de tarefas do Windows é o `ionflow.ico` | | |

### T-G-002 — Sidebar: botões e controles

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 2.1 | Botão **Importar EIS para raw** visível e alinhado | | |
| 2.2 | Botão **Importar Ciclagem para processed** visível | | |
| 2.3 | Botão **Gráficos Interativos** visível | | |
| 2.4 | Botão **Rodar Pipeline EIS** visível | | |
| 2.5 | Botão **Rodar Pipeline Ciclagem** visível | | |
| 2.6 | Botão **Rodar Ambos** visível | | |
| 2.7 | Botão **Rodar Pipeline DRT** visível | | |
| 2.8 | Frame DRT (lambda, N, preset) visível, campos legíveis | | |
| 2.9 | Botão **Gerar Relatório PDF** visível | | |
| 2.10 | Botão **Exportar EIS como…** visível | | |
| 2.11 | Botão **Comparar Amostras** visível | | |
| 2.12 | Botão **Batch Processing** visível | | |
| 2.13 | Botão **Auto-Compor Circuitos** visível | | |
| 2.14 | Botão **Salvar Configuração** visível | | |
| 2.15 | **Selector Provedor IA** (Segmented: Nenhum/OpenAI/Ollama) visível | | |
| 2.16 | Label **Status: pronto** visível na parte inferior | | |
| 2.17 | Barra de progresso visível (não animando no idle) | | |
| 2.18 | Selector **Tema** (Claro/Escuro/Sistema) visível | | |
| 2.19 | Selector **Idioma** (Português/English/Español) visível | | |
| 2.20 | **Botão 📚 Referências** visível, com borda outline, sem sobreposição | | |
| 2.21 | Nenhum botão está cortado ou fora da sidebar | | |
| 2.22 | Resize da janela: sidebar mantém largura mínima usável | | |

---

## BLOCO 2 — GUI: Abas (Tabs)

### T-G-003 — Aba "Gráficos"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 3.1 | Aba renderiza sem erro; área de gráfico inicialmente vazia | | |
| 3.2 | Após rodar pipeline EIS: gráfico Nyquist aparece corretamente | | |
| 3.3 | Gráfico não sobrepõe bordas da aba | | |
| 3.4 | Eixos com labels legíveis (Z', -Z'', escala correta) | | |
| 3.5 | Legenda do gráfico visível e sem corte | | |

### T-G-004 — Aba "Tabelas"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 4.1 | Aba renderiza sem erro; tabela vazia se nenhum pipeline rodou | | |
| 4.2 | Após pipeline EIS: tabela exibe colunas (Sample, Rs, Rp, …) | | |
| 4.3 | Scroll horizontal funciona se tabela ultrapassar largura | | |
| 4.4 | Scroll vertical funciona para muitas linhas | | |
| 4.5 | Seleção de linha funciona sem erro visual | | |

### T-G-005 — Aba "Logs"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 5.1 | Textbox de logs visível, fundo escuro | | |
| 5.2 | Mensagens de log aparecem em tempo real durante pipeline | | |
| 5.3 | Scroll automático para última linha funciona | | |
| 5.4 | Texto não sai das bordas do widget | | |

### T-G-006 — Aba "🤖 Análise IA"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 6.1 | Checkboxes EIS/Ciclagem/DRT visíveis e alinhados | | |
| 6.2 | Botão segmentado summary/full visível | | |
| 6.3 | Botão "Executar Análise IA" alinhado à direita | | |
| 6.4 | Textbox de resultado visível e redimensionável com a janela | | |

### T-G-007 — Aba "🔬 Validação KK"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 7.1 | Botão "Executar Validação Kramers-Kronig" visível | | |
| 7.2 | Label descritivo visível ao lado do botão | | |
| 7.3 | Textbox de resultado ocupa área restante | | |

### T-G-008 — Aba "🩺 Diagnóstico Fitting"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 8.1 | Aba renderiza sem erro | | |
| 8.2 | Controles/resultados de diagnóstico visíveis | | |

### T-G-009 — Aba "📝 Relatório Fitting"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 9.1 | Textbox de relatório visível | | |
| 9.2 | Após pipeline: conteúdo aparece formatado (não HTML bruto) | | |

### T-G-010 — Aba "🔄 Comparar Amostras"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 10.1 | Aba renderiza sem erro | | |
| 10.2 | Controles de seleção de amostras visíveis | | |
| 10.3 | Gráfico comparativo aparece após selecionar amostras | | |

### T-G-011 — Aba "⚙️ Configurações"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 11.1 | Aba renderiza sem erro | | |
| 11.2 | Todos os campos de configuração visíveis | | |
| 11.3 | Campos de texto não cortados | | |

---

## BLOCO 3 — GUI: Janelas Secundárias (CTkToplevel)

### T-G-012 — Janela "Gráficos Interativos"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 12.1 | Clique em "Gráficos Interativos" abre nova janela sem erro | | |
| 12.2 | Título da janela correto | | |
| 12.3 | Gráficos renderizados sem sobreposição de elementos | | |
| 12.4 | Segunda abertura: reutiliza janela existente (não abre duplicata) | | |
| 12.5 | Fechar janela não causa crash na janela principal | | |

### T-G-013 — Janela "📚 Referências" *(nova)*

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 13.1 | Clique em "📚 Referências" abre CTkToplevel | | |
| 13.2 | Título: `Referências Bibliográficas — IonFlow Pipeline v0.3.0` | | |
| 13.3 | Textbox carregada com texto (não vazio) na abertura | | |
| 13.4 | Referências exibidas em seções (cabeçalhos coloridos) | | |
| 13.5 | Tags `[EIS-1]`, `[DRT-1]` etc. em cor diferente (amarelo) | | |
| 13.6 | Linhas `→` em verde | | |
| 13.7 | DOIs sublinhados em azul | | |
| 13.8 | Clicar em DOI abre `https://doi.org/…` no browser padrão | | |
| 13.9 | Cursor muda para `hand2` ao passar sobre DOI | | |
| 13.10 | Campo de busca presente no topo | | |
| 13.11 | Digitar "DRT" filtra apenas refs DRT (contador atualiza) | | |
| 13.12 | Digitar texto inexistente mostra mensagem "(Nenhuma…)" | | |
| 13.13 | Limpar campo de busca restaura todas as 56 refs | | |
| 13.14 | Botão "Limpar" limpa o campo | | |
| 13.15 | Botão "Fechar" fecha a janela sem crash | | |
| 13.16 | Segunda abertura: reutiliza janela existente (lift + focus) | | |
| 13.17 | Scroll vertical funciona em toda a lista | | |
| 13.18 | Redimensionar janela: textbox expande corretamente | | |

### T-G-014 — Dialog "Atualização disponível"

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 14.1 | Se versão local < remota: dialog abre corretamente | | |
| 14.2 | Se versão está atualizada: dialog não aparece | | |
| 14.3 | Botões "Baixar" e "Cancelar" visíveis e funcionais | | |

---

## BLOCO 4 — GUI: Temas e Idiomas

### T-G-015 — Troca de tema

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 15.1 | Mudar para **Claro**: toda UI muda para fundo branco/cinza | | |
| 15.2 | Mudar para **Sistema**: segue preferência do SO | | |
| 15.3 | Mudar para **Escuro**: retorna ao tema escuro | | |
| 15.4 | Janela de Referências aberta acompanha a troca de tema | | |
| 15.5 | Sem sobreposição de cores após mudança | | |

### T-G-016 — Troca de idioma

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 16.1 | Mudar para **English**: botões da sidebar traduzidos | | |
| 16.2 | Mudar para **Español**: botões da sidebar traduzidos | | |
| 16.3 | Mudar para **Português**: volta ao PT-BR | | |
| 16.4 | Labels de aba traduzidos corretamente | | |
| 16.5 | Nenhum label fica em `tr("…")` literal (função não resolvida) | | |
| 16.6 | Status bar traduzida | | |

---

## BLOCO 5 — GUI: Fluxos de Pipeline

### T-G-017 — Pipeline EIS (caminho feliz)

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 17.1 | Importar arquivo `.csv` EIS: sem erro no terminal | | |
| 17.2 | Clicar "Rodar Pipeline EIS": barra de progresso anima | | |
| 17.3 | Status muda para "Processando…" durante execução | | |
| 17.4 | Status volta para "Pronto" após conclusão | | |
| 17.5 | Gráfico Nyquist aparece na aba Gráficos | | |
| 17.6 | Tabela de resultados populada na aba Tabelas | | |
| 17.7 | Nenhum traceback no Log | | |

### T-G-018 — Pipeline com erro de arquivo inválido

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 18.1 | Importar arquivo corrompido/vazio: mensagem de erro amigável | | |
| 18.2 | GUI não trava (botões respondem após o erro) | | |
| 18.3 | Status bar mostra mensagem de erro clara | | |

### T-G-019 — Botão Referências durante pipeline em execução

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 19.1 | Abrir janela de Referências enquanto pipeline roda: sem deadlock | | |
| 19.2 | Pipeline termina normalmente após fechar a janela | | |

---

## BLOCO 6 — Dashboard: Sidebar e Navegação

### T-D-001 — Inicialização

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 1.1 | Página carrega em `http://localhost:8501` sem erro 500 | | |
| 1.2 | Sidebar exibe logo `⚡ IonFlow` e versão | | |
| 1.3 | Todas as 8 páginas listadas no sidebar: Visão Geral, Upload & Dados, Resultados EIS, Análise DRT, Ciclagem, Histórico ML, Análise IA, **Referências** | | |
| 1.4 | Página inicial padrão: 🏠 Visão Geral | | |
| 1.5 | Caption `DB: ionflow.db` visível no rodapé do sidebar | | |

### T-D-002 — Navegação entre páginas

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 2.1 | Clicar em cada página: conteúdo correto carrega sem erro | | |
| 2.2 | Sem "AttributeError" ou "KeyError" em nenhuma página | | |
| 2.3 | Voltar para Visão Geral: métricas exibidas | | |
| 2.4 | URL não muda para 404 em nenhuma página | | |

---

## BLOCO 7 — Dashboard: Páginas Individuais

### T-D-003 — 🏠 Visão Geral

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 3.1 | Título `🏠 Visão Geral` visível | | |
| 3.2 | Cards de métricas (total EIS, campanhas, etc.) visíveis | | |
| 3.3 | Métricas são números, não `None` ou `NaN` | | |
| 3.4 | Seção de gráfico de overview renderiza | | |

### T-D-004 — 📤 Upload & Dados

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 4.1 | Widget de upload de arquivo visível | | |
| 4.2 | Upload de CSV EIS válido: preview da tabela aparece | | |
| 4.3 | Upload de arquivo inválido: mensagem de erro amigável | | |
| 4.4 | Botão de confirmar/processar visível | | |

### T-D-005 — 📊 Resultados EIS

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 5.1 | Página carrega sem erro mesmo com DB vazio | | |
| 5.2 | Gráfico Nyquist renderiza quando há dados | | |
| 5.3 | Selector de amostras funciona | | |
| 5.4 | Tabela de parâmetros EIS (Rs, Rp, C…) visível | | |
| 5.5 | Gráfico não ultrapassa margem da página | | |

### T-D-006 — 🌀 Análise DRT

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 6.1 | Página carrega sem erro | | |
| 6.2 | Gráfico DRT (γ vs τ) renderiza com dados | | |
| 6.3 | Picos marcados no gráfico com labels legíveis | | |
| 6.4 | Tabela de picos DRT visível | | |

### T-D-007 — 🔋 Ciclagem

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 7.1 | Página carrega sem erro | | |
| 7.2 | Gráfico de capacitância vs ciclo renderiza | | |
| 7.3 | Diagrama de Ragone visível (se dados disponíveis) | | |
| 7.4 | Métricas (C_grav, E, P) exibidas corretamente | | |

### T-D-008 — 🗄️ Histórico ML

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 8.1 | Página carrega sem erro | | |
| 8.2 | Tabela de histórico de experimentos visível | | |
| 8.3 | Filtros funcionam sem erro | | |
| 8.4 | Health Score exibido com cor/indicador visual | | |

### T-D-009 — 🤖 Análise IA

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 9.1 | Selector de provedor (none/ollama/openai) visível | | |
| 9.2 | Com `none` selecionado: botão desabilitado, info box aparece | | |
| 9.3 | Campo de contexto (textarea) visível e redimensionável | | |
| 9.4 | Campo de instrução adicional visível | | |

### T-D-010 — 📚 Referências *(nova página)*

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 10.1 | Título `📚 Referências Bibliográficas` visível | | |
| 10.2 | Caption com versão v0.3.0 visível | | |
| 10.3 | Campo de filtro visível (placeholder: `ex: EIS, DRT…`) | | |
| 10.4 | Counter `56 / 56 referências` visível inicialmente | | |
| 10.5 | Divisor horizontal visível abaixo do counter | | |
| 10.6 | Seções expandíveis (expanders) para cada PARTE | | |
| 10.7 | Expanders abertos por padrão (sem texto de filtro) | | |
| 10.8 | Expanders fechados por padrão quando filtro está ativo | | |
| 10.9 | Cada referência exibe texto completo dentro do expander | | |
| 10.10 | DOIs aparecem como links markdown clicáveis `[10.xxxx](https://doi.org/…)` | | |
| 10.11 | Linhas `→` aparecem em itálico | | |
| 10.12 | Digitar "EIS" no filtro: apenas seções EIS visíveis, counter atualiza | | |
| 10.13 | Digitar "Randles" filtra apenas a ref FIT-5 | | |
| 10.14 | Digitar texto sem match: mensagem "Nenhuma referência encontrada…" | | |
| 10.15 | Limpar campo: todas as 56 refs restauradas, expanders reabertos | | |
| 10.16 | Clicar em DOI: abre tab no browser (não erro 404) | | |
| 10.17 | Scrollbar vertical funciona na página inteira | | |
| 10.18 | Layout responsivo: reduzir janela do browser não quebra layout | | |

---

## BLOCO 8 — Dashboard: Temas e Responsividade

### T-D-011 — Responsividade

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 11.1 | Largura 1920px: sem espaço em branco excessivo | | |
| 11.2 | Largura 1280px: conteúdo principal não cortado | | |
| 11.3 | Largura 768px (tablet): sidebar colapsa ou scrollável | | |
| 11.4 | Sem scroll horizontal na área de conteúdo principal | | |

### T-D-012 — Modo escuro do Streamlit

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 12.1 | Ativar modo escuro via `⋮ > Settings > Theme`: UI muda | | |
| 12.2 | Gráficos matplotlib legíveis no modo escuro | | |
| 12.3 | Tabelas com contraste suficiente no modo escuro | | |

---

## BLOCO 9 — Testes de Regressão Visual

### T-R-001 — Consistência entre GUI e Dashboard

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 1.1 | Valores de Rs, Rp, C na GUI batem com os do Dashboard | | |
| 1.2 | Gráfico Nyquist visualmente idêntico em ambas as interfaces | | |
| 1.3 | Mesmos nomes de amostras exibidos em ambas | | |

### T-R-002 — Estado persistido ao reiniciar

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 2.1 | Fechar e reabrir GUI: preferências de idioma e tema restauradas | | |
| 2.2 | Recarregar dashboard: dados do DB carregados novamente | | |

---

## BLOCO 10 — Casos Extremos (Edge Cases)

### T-E-001 — Arquivos de entrada problemáticos

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 1.1 | Arquivo EIS com 0 linhas de dados: erro amigável, sem crash | | |
| 1.2 | Arquivo EIS com separador errado (`;` em vez de `,`): erro claro | | |
| 1.3 | Arquivo com caracteres especiais no nome (`ação.csv`): sem crash | | |
| 1.4 | Dois pipelines rodados em sequência: resultados não se misturam | | |

### T-E-002 — Janela de Referências com filtros extremos

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 2.1 | Digitar 200+ caracteres no campo de filtro: sem crash | | |
| 2.2 | Digitar caracteres especiais (`<script>`, `%`, `\n`): sem crash | | |
| 2.3 | Clicar muito rapidamente em múltiplos DOIs: nenhum hang | | |

### T-E-003 — Dashboard com DB vazio

| # | Verificação | Pass | Defeito |
|---|-------------|------|---------|
| 3.1 | Iniciar dashboard sem `data/ionflow.db`: sem crash (cria vazio) | | |
| 3.2 | Todas as páginas navegáveis mesmo sem dados | | |
| 3.3 | Mensagens de "sem dados" amigáveis em vez de stacktraces | | |

---

## Log de Defeitos

> Preencha durante os testes. Use o formato abaixo.

```
ID: DEF-001
Bloco/Teste: T-G-013 / item 13.7
Severidade: Alta | Média | Baixa
Componente: GUI - Janela Referências
Descrição: DOIs não aparecem sublinhados no tema Claro.
Reprodução: 1) Abrir Referências; 2) Mudar tema para Claro.
Screenshot: (path)
Status: Aberto
```

---

## Critérios de Aprovação para Release

| Categoria | Critério |
|-----------|----------|
| **Bloqueadores** | Zero defeitos de Severidade Alta (crashes, dados errados, funcionalidade core inacessível) |
| **Visuais** | Máximo 3 defeitos Médios registrados com workaround documentado |
| **Edge cases** | Todos os T-E-001 passando (sem crash com entradas inválidas) |
| **Novas features** | T-G-013 (Referências GUI) e T-D-010 (Referências Dashboard) 100% passando |

---

## Checklist Final de Release

- [ ] Todos os bloqueadores corrigidos e re-testados
- [ ] `python -m pytest tests/ -q` passa sem falhas
- [ ] `python -c "import ast; ast.parse(open('gui_app.py').read())"` sem SyntaxError
- [ ] `python -c "import ast; ast.parse(open('dashboard.py').read())"` sem SyntaxError
- [ ] Nenhum `print()` de debug esquecido no código
- [ ] Commit limpo com mensagem `test: visual QA v0.3.0 approved`
