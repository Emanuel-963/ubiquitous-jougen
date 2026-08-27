# IonFlow Pipeline

**Plataforma profissional de análise eletroquímica** — EIS, ciclagem galvanostática, DRT — com agente IA, relatórios PDF, CLI e GUI interativa.

[![CI](https://github.com/Emanuel-963/ubiquitous-jougen/actions/workflows/ci.yml/badge.svg)](https://github.com/Emanuel-963/ubiquitous-jougen/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.5.1-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-2263%2B%20passing-brightgreen)

---

## ⚡ Quick Start

```bash
# 1. Instalar
python -m venv venv && venv\Scripts\Activate.ps1
pip install -e .

# 2. GUI
python gui_app.py

# 3. CLI — pré-processar e analisar (workflow metrológico recomendado)
ionflow preprocess --data-dir data/raw --output data/clean
ionflow eis --data-dir data/clean --output outputs/
ionflow analyze --all --ai --export-pdf report.pdf
```

### Perfis benchmark por config (GUI/CLI)

- Exemplo pronto: `docs/config_benchmark_profiles.example.json`
- Para aplicar no CLI: `ionflow config load --path docs/config_benchmark_profiles.example.json`
- Na GUI: carregar o mesmo arquivo em **Config > Load** para usar os objetivos customizados no tab **Lab Intelligence**.

---

## ✨ Funcionalidades (v0.5.1)

> **Legenda de maturidade:**  
> 🟢 **Estável** — testado, comportamento previsível, adequado para produção científica.  
> 🟡 **Parcialmente validado** — funciona, mas validação quantitativa vs. referência externa ainda incompleta.  
> 🔴 **Experimental** — funcionalidade existe, pouco ou nenhum teste de precisão; usar com cautela.

| Módulo | Descrição | Status |
|---|---|:---:|
| **Pipeline EIS** | 37 circuitos equivalentes, ML shortlist, Monte Carlo, Kramers-Kronig | 🟡 |
| **Metrologia Orazem** | σ = α\|Zⱼ\|+β\|Zᵣ\|, χ²/ν, IC 95%, parâmetro-zumbi, Porous-TLM | 🟡 |
| **Pipeline DRT** | Tikhonov regularisation, detecção de picos, overlay multi-amostra | 🟡 |
| **Pipeline Ciclagem** | Ragone com zonas de referência, gap analysis vs targets | 🟡 |
| **Kramers-Kronig** | Diagnóstico de linearidade/estabilidade (método Boukamp) | 🟡 |
| **Incerteza paramétrica** | Monte Carlo + Bootstrap sobre parâmetros do fitting | 🟡 |
| **Importação nativa** | Parsers Gamry (.dta), BioLogic (.mpr/.mpt), Autolab, Zahner (.isc) | 🟡 |
| **Exportação científica** | ZView, LaTeX booktabs, OriginPro, MEISP | 🟢 |
| **GUI** | Workspace V3, wizard guiado, Ribbon contextual, Command Palette (Ctrl+K), Inspector com rolagem, 3 idiomas | 🟢 |
| **CLI** | `ionflow eis / cycling / drt / analyze / preprocess / validate / config` | 🟢 |
| **📄 PDF Reports** | Relatório automático com capa, secções EIS/Ciclagem/DRT/IA | 🟢 |
| **Batch** | Processamento paralelo com `ProcessPoolExecutor` | 🟢 |
| **Análise comparativa** | Overlay N amostras, Health Score (0–10), PCA automático | 🔴 |
| **🤖 Agente IA** | 50+ regras eletroquímicas + LLM generativo (OpenAI / Ollama) | 🔴 |
| **Classificador ML** | RandomForest shortlist de circuito (~62 % acc. hierárquico) | 🔴 |
| **Predição de performance** | ML para energia/potência/retenção (treinado em dados sintéticos) | 🔴 |
| **Base de dados** | SQLite local: amostras, EIS, DRT, ciclagem, histórico ML | 🟢 |
| **Dashboard web** | Streamlit — 7 páginas em `localhost:8501` | 🔴 |
| **Auto-update** | Verificação e download do instalador Windows pelo GitHub | 🟡 |

---

## 🔬 Para Pesquisadores

### Fluxo típico de análise (v0.5.1 — estilo Orazem & Tribollet 2026)

1. **Iniciar projeto** — No Workspace, abra o wizard e escolha um preset por objetivo.
2. **Preparar dados** — Colocar ficheiros EIS (`.csv`/`.txt`) em `data/raw/` e ciclagem em `data/processed/`
3. **Pré-processar** (recomendado):
   ```bash
   ionflow preprocess --data-dir data/raw --output data/clean
   ```
   Remove ponto HF, filtra 50/100 Hz, opcionalmente trunca em fc.
4. **Executar** — `python gui_app.py` ou `ionflow eis --data-dir data/clean`
5. **Pipeline automático:**
   - Carregamento → Validação (KK + powerline check) → Fitting ponderado por σ(f)
   - χ²/ν + IC 95% por parâmetro → Ranking por BIC → PCA → Heatmaps
6. **Acompanhar** — A barra mostra percentual e etapa atual; detalhes continuam na aba Logs.
7. **Análise IA** — Clique em "🤖 Análise IA", use `Ctrl+K` ou `ionflow analyze --all --ai`
8. **Exportar** — PDF com 1 clique, ou `ionflow analyze --export-pdf report.pdf`

### Métricas extraídas

- **Rs** (resistência série), **Rp** (resistência de polarização)
- **CPE** (n, Q), **Sigma** (Warburg), **Tau** (constante de tempo)
- **Capacitância efetiva**, **Energia armazenada**, **Dispersion Index**
- **χ²/ν** (goodness-of-fit ponderado pela estrutura de erro estocástico)
- **IC 95%** por parâmetro, **significância estatística** (significativo / zumbi)
- **Incerteza:** Monte Carlo Orazem-weighted (N=100) + Bootstrap

### Rigor metrológico (Tribollet & Orazem 2026)

O pipeline implementa a metodologia de [Tribollet & Orazem, *Electrochimica Acta* 568, 149009 (2026)](https://doi.org/10.1016/j.electacta.2026.149009):

- **Estrutura de erro**: σ(f) = 0.001216·|Zⱼ| + 0.000333·|Zᵣ| (Eq. 9 do artigo)
- **Remoção de ponto HF**: elimina transitório de comutação
- **Filtro 50/100 Hz**: remove ruído de rede elétrica inadequadamente filtrado
- **Truncamento em fc**: elimina influência da impedância ôhmica acima de fc = 1/(2πRe·C∞)
- **Circuito Porous-Coating-TLM**: eletrodo poroso (TLM) quando n ≈ 0.5 (Eqs. 16–18)
- **Tutorial 24**: [tutoriais/24_metrologia_orazem_tribollet.txt](tutoriais/24_metrologia_orazem_tribollet.txt)

---

## 🤖 Agente IA — Exemplos

```
📊 Resumo Executivo
"A amostra Nb₂/H₂SO₄ apresenta Rs=2.66Ω (baixo) e n=0.77 (rugosidade moderada).
O circuito Randles-CPE-W foi selecionado com 78% de confiança (BIC=45.2)."

⚠️ Anomalias
• Rs do Na₂SO₄ é 4.5× maior que amostras em H₂SO₄
• Rp convergiu para o limite superior — possível não-convergência

💡 Recomendações
1. [ALTA] Polir eletrodo ou usar cola de prata para reduzir Rs
2. [MÉDIA] Expandir faixa de frequência para 10 mHz–1 MHz
3. [MÉDIA] H₂SO₄ proporciona Rs 78% menor → priorizar como eletrólito

🔮 Previsões
• Energia estimada: 12 ± 3 μJ
• Retenção estimada: 85 ± 8%
```

---

## 🧪 Protocolo Científico Multicritério (v0.5.1)

O módulo independente `src/scientific_protocol/` automatiza o protocolo
experimental multicritério usando como referência primária os quatro scripts
em `Scripts Adicionais/` e a planilha
`Classificacao_Melhor_Eletrolito_IonFlow_atualizada.xlsx`.

### Configuração pelo GUI

No GUI principal, abra **Ferramentas > Protocolo Científico Multicritério**
ou use `Ctrl+K` e pesquise esse comando. Escolha **Nova análise** para abrir o
wizard gráfico. O wizard guia a identificação do projeto, arquivos CV/GCD/EIS,
massas, área, eletrodos e sequência GCD. Ao final, use **Salvar e executar** ou
**Salvar sem executar**.

Para continuar um trabalho, escolha **Carregar análise existente**. As células
e replicatas podem ser adicionadas, duplicadas sem copiar arquivos experimentais
ou excluídas após confirmação. O arquivo `project.json` é criado internamente;
o pesquisador não precisa editá-lo.

O formato novo usa `schema_version: "2.0"`, agrupa células por eletrólito e
guarda metadados de reprodutibilidade. Arquivos antigos no formato simples de
`electrolytes` continuam sendo carregados e migrados automaticamente para uma
célula `cell_001`.
Ele reutiliza as rotinas validadas para:

- CV/Dunn: capacitância específica, `b-value` e contribuição capacitiva;
- GCD: `Cs`, eficiência coulômbica, retenção, energia, potência e `IR drop`;
- EIS: `Rs`, `Rct/Rp`, Nyquist e ajuste equivalente nos estados inicial,
  pós-CV e final;
- DRT: curva-L, picos confiáveis e `area_drt_lenta` para `tau >= 0.1 s`.

O ranking segue exatamente os pesos da planilha: `20/15/10/15/10/5/10/10/5`,
com normalização min-max, pesos renormalizados para métricas disponíveis,
`N/D` preservado como ausente, cobertura e status `Completo`/`Parcial`.

### Entrada recomendada

Use um arquivo `protocol.json` em `data_dir`; há um modelo em
[docs/scientific_protocol_example.json](docs/scientific_protocol_example.json).
Os caminhos podem ser relativos ao diretório que contém o manifesto. Para cada
eletrólito, informe `cv`, `gcd`, `eis`, `mass_g` e `current_sequence_a_g` quando
esses ensaios existirem. A retenção GCD fica `N/D` quando não há bloco de
retorno à mesma corrente.

### Uso

```python
from src.scientific_protocol import run_scientific_protocol

result = run_scientific_protocol(
   data_dir="dados_experimentais",
   output_dir="results",
   config="default",
)
print(result["ranking"][["electrolyte", "score_available", "coverage_pct", "protocol_status"]])
print(result["summary"])
```

As saídas são organizadas em `results/cv`, `results/gcd`, `results/eis`,
`results/drt`, `results/ranking` e `results/report`, incluindo PNGs, CSVs,
`classificacao_multicriterio.csv`, `classificacao.png`,
`metricas_completas.json` e `resumo_cientifico.txt`.

O módulo permanece desacoplado da GUI e dos pipelines existentes. Resultados
DRT com picos lentos não confiáveis são tratados como ausentes no ranking, não
como zero; a interpretação final ainda requer validação experimental.

---

## 📁 Estrutura do Projeto

```
eis_analytics/
├── gui_app.py                # GUI principal
├── main.py / main_cycling.py / main_drt.py
├── build_exe.py              # Build PyInstaller
├── pyproject.toml            # Metadados + dependências
├── src/                      # 35+ módulos
│   ├── config.py             # PipelineConfig centralizado
│   ├── models.py             # Dataclasses tipadas (EISResult, etc.)
│   ├── validation.py         # Validação de entrada + KK
│   ├── kramers_kronig.py     # Teste Kramers-Kronig (Boukamp)
│   ├── circuit_registry.py   # 7 circuitos extensíveis
│   ├── circuit_composer.py   # Recombinação automática
│   ├── ml_circuit_selector.py# ML shortlist (RandomForest)
│   ├── fitting_diagnostics.py# Diagnóstico visual 🟢🟡🔴
│   ├── fitting_report.py     # Feedback textual
│   ├── uncertainty.py        # Monte Carlo + Bootstrap
│   ├── report_generator.py   # PDF automático (fpdf2)
│   ├── batch_processor.py    # Processamento paralelo
│   ├── cli.py                # CLI profissional
│   ├── i18n.py               # 3 idiomas (PT/EN/ES)
│   ├── ai/                   # Agente IA
│   │   ├── knowledge_base.py # 50+ regras eletroquímicas
│   │   ├── inference_engine.py
│   │   ├── performance_predictor.py
│   │   ├── process_advisor.py
│   │   └── llm_adapter.py    # LLM opcional (OpenAI/Ollama)
│   └── gui/                  # GUI modular (MVC)
│       ├── controller.py / models.py / main_window.py
│       ├── shortcuts.py / widgets.py
│       └── tabs/ (eis, cycling, drt, advanced, tables, ai_panel)
├── data/knowledge/           # Regras eletroquímicas (JSON)
├── tests/                    # 1782 testes automatizados
├── themes/                   # Tema visual da GUI
└── outputs/                  # Tabelas, figuras, Excel, PDF
```

---

## 🚀 Instalação

Escolha **uma** das três formas abaixo.

---

### Opção 1 — Instalador Windows (recomendado para pesquisadores)

> Não requer Python, Git nem terminal.  O auto-update cuida de versões futuras.

1. Acesse a última versão em <https://github.com/Emanuel-963/ubiquitous-jougen/releases/latest>
2. Para a versão atual, baixe [`IonFlow_Pipeline_Setup_0.5.1.exe`](https://github.com/Emanuel-963/ubiquitous-jougen/releases/download/v0.5.1/IonFlow_Pipeline_Setup_0.5.1.exe)
3. Execute o instalador e siga os passos (não requer permissão de administrador)
4. Abra pelo atalho na área de trabalho ou pelo menu Iniciar

**Atualizar para nova versão:**  
A GUI verifica atualizações ao iniciar.  Quando aparecer a notificação:
- Clique **"⬇ Instalar agora (automático)"**
- O instalador baixa, substitui os arquivos em segundo plano e reabre o IonFlow
- Seus dados em `data/raw/` e `data/processed/` não são tocados

---

### Opção 2 — Git clone (para desenvolvedores e usuários avançados)

**Pré-requisitos:**
- Python 3.11+ — <https://www.python.org/downloads/> (marcar "Add to PATH")
- Git — <https://git-scm.com/download/win> (Windows) ou `sudo apt install git` (Linux)

**Instalação:**

```powershell
# 1. Clonar o repositório
git clone https://github.com/Emanuel-963/ubiquitous-jougen.git
cd ubiquitous-jougen/eis_analytics

# 2. Criar e ativar ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate           # Linux / macOS

# 3. Instalar o pacote
pip install -e .

# 4. Abrir a GUI
python gui_app.py
```

Ao abrir pela primeira vez, use **Iniciar wizard de projeto** no Workspace. Os
presets disponíveis são: Triagem rápida, Diagnóstico EIS, Desempenho de
Ciclagem, Caracterização Completa, Validação DRT e Relatório publicável.

> **Nota Windows:** Se o PowerShell bloquear a ativação, rode primeiro:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

**Atualizar para nova versão (git clone):**

```powershell
# Na pasta eis_analytics, com o venv ativado:
git pull
pip install -e .
python gui_app.py
```

Ou use o script automático:

```powershell
.\install_pipeline.ps1 -Update
```

**Verificar se está tudo OK:**

```powershell
 python -c "import src; print(src.__version__)"   # deve mostrar 0.5.1 ou superior
python -m pytest tests/ -q --tb=no               # deve passar todos os testes
```

---

### Opção 3 — Instalação via script automático (Windows)

O script `install_pipeline.ps1` configura tudo automaticamente:

```powershell
# Primeira instalação
.\install_pipeline.ps1

# Atualizar versão existente
.\install_pipeline.ps1 -Update
```

O script:
- Verifica/instala Python 3.11
- Cria/recria o ambiente virtual
- Instala todas as dependências
- Cria as pastas `data/raw/`, `data/processed/`, `outputs/`
- Com `-Update`: faz `git pull` + reinstala dependências

---

### Estrutura de pastas após instalação

```
eis_analytics/
├── data/
│   ├── raw/          ← Coloque aqui seus arquivos EIS (.txt, .csv, .dta, .mpt)
│   └── processed/    ← Coloque aqui seus dados de ciclagem
├── outputs/          ← Resultados gerados automaticamente
└── IonFlow_Pipeline.exe  (se instalador) / gui_app.py (se git clone)
```

---

## ▶️ Uso

### GUI

```bash
ionflow                 # via entry-point
python gui_app.py       # direto
```

### CLI

```bash
ionflow-cli eis --data-dir data/raw --output outputs/
ionflow-cli cycling --data-dir data/processed --scan-rate 0.1
ionflow-cli drt --data-dir data/raw --lambda 1e-3
ionflow-cli analyze --all --ai --export-pdf report.pdf
ionflow-cli validate --data-dir data/raw
ionflow-cli config --init
```

### Scripts de pipeline

```bash
python main.py          # Pipeline EIS completo
python main_cycling.py  # Pipeline ciclagem
python main_drt.py      # Pipeline DRT
```

---

## 🧪 Desenvolvimento

```bash
pip install -e ".[dev]"
pytest -q                # 2263+ testes
black . && isort .       # formatação
flake8 .                 # linting
mypy src                 # type checking
```

---

## 📊 Métricas v0.1.0 → v0.2.0 → v0.3.0

| Métrica | v0.1.0 | v0.2.0 | v0.3.0 |
|---------|--------|--------|--------|
| Módulos Python | 18 | 35+ | **50+** |
| Circuitos | 3 | 7 | **7** (extensível) |
| Testes automatizados | ~70% cov | 1782 | **220+ (novos)** |
| Idiomas | 2 (PT/EN) | 3 (+ ES) | **3** |
| Agente IA | ❌ | ✅ Rule-based + ML | ✅ **+ LLM generativo** |
| PDF Reports | ❌ | ✅ | ✅ |
| CLI | ❌ | ✅ | ✅ |
| Importação nativa | ❌ | ❌ | ✅ **Gamry / BioLogic / Autolab / Zahner** |
| Exportação científica | ❌ | ❌ | ✅ **ZView / LaTeX / Origin / MEISP** |
| Health Score + PCA | ❌ | ❌ | ✅ |
| SQLite + Dashboard | ❌ | ❌ | ✅ **Streamlit localhost:8501** |
| Auto-update | ❌ | ❌ | ✅ |

---

## ⚠️ Limitações Conhecidas e Estado de Validação

> **Beta estável** — pipeline principal (EIS / ciclagem / DRT) testado e funcional.  
> Adequado para investigação científica; ainda **não** submetido a revisão por pares.  
> Ver [docs/SCIENTIFIC_VALIDATION.md](docs/SCIENTIFIC_VALIDATION.md) para auditoria detalhada módulo a módulo.

### Limitações críticas a conhecer antes de publicar resultados

| Área | Limitação |
|------|-----------|
| **Validação vs. referência externa** | Nenhum módulo foi ainda benchmarkado contra dados experimentais publicados ou software de referência (ZView, EC-Lab, pyDRTtools). Todos os testes automáticos usam dados sintéticos gerados internamente. |
| **Fitting paramétrico** | A precisão de recuperação de Rs, Rp, Q, n não está quantificada. Use sempre o χ²/ν e os IC 95 % como guia; verifique o ajuste visualmente. |
| **Coeficientes Orazem α/β** | Calibrados para sistemas aquosos em temperatura ambiente (Tribollet & Orazem 2026). Não usar diretamente em meios não-aquosos, temperaturas extremas ou células com geometria não-plana sem re-calibração. |
| **DRT regularização** | λ de Tikhonov fixo ou ajustado heuristicamente. A escolha de λ afeta número e posição dos picos. Adequado para análise qualitativa; não use valores absolutos de γ(τ) como referência quantitativa sem inspeção da L-curve. |
| **Classificador ML de circuito** | Acurácia real: ~30 % (flat, 37 classes), ~62 % (hierárquico por família). Treinado em espectros sintéticos. **Tratar sempre como sugestão** — confirmar o circuito manualmente. |
| **Predição de performance (ML ciclagem)** | Treinado exclusivamente em dados sintéticos; sem validação experimental. Valores de energia/potência preditos são **estimativas heurísticas**, não previsões calibradas. |
| **Agente IA (regras + LLM)** | As 50+ regras eletroquímicas são heurísticas de domínio, não derivadas de modelos termodinâmicos. O LLM generativo (OpenAI/Ollama) pode alucinar — **nunca citar output do LLM diretamente** num paper. |
| **PCA automático** | Não benchmarkado contra sklearn PCA; loadings/scores não validados. Usar apenas para exploração visual de agrupamentos. |
| **Parser BioLogic `.mpr`** | Suporte limitado a versões de firmware testadas internamente. Arquivos de firmware mais recentes podem falhar silenciosamente. |
| **Plataformas** | Testado em Windows 10/11. Linux/macOS via `install_linux.sh` (experimental — sem CI dedicado). |
| **GPU** | Não utilizado. Todos os cálculos em CPU. |

### O que está bem validado

| Módulo | Evidência |
|--------|-----------|
| **Fórmula de energia armazenada** E = ½CV² | Teste unitário com `math.isclose` vs valor analítico |
| **Kramers-Kronig (comportamento qualitativo)** | 20+ testes: RC ideal → "excelente", dados ruidosos → "suspeito" |
| **Monte Carlo de incerteza (convergência)** | Testes verificam média próxima do verdadeiro e IC ordenados |
| **DRT — não-negatividade e monotonia de τ** | Testes explícitos de invariante físico |
| **Pipeline EIS end-to-end** | Teste de integração: carga → KK → fitting → relatório sem erros |

---

## 📄 Licença

MIT — veja [LICENSE](LICENSE).

## 🎓 Contexto Acadêmico

Projeto desenvolvido para análise de materiais eletroquímicos em contexto de investigação científica.

## 📚 Documentação

- [CHANGELOG](CHANGELOG.md) — Histórico de alterações
- [ONE_PAGER](docs/ONE_PAGER.md) — Resumo de 1 página
- [PRESENTATION](docs/PRESENTATION.md) — Guia de apresentação
- [ENTREGA_PROFESSOR](docs/ENTREGA_PROFESSOR.md) — Documento de entrega ao orientador
- [SCIENTIFIC_VALIDATION](docs/SCIENTIFIC_VALIDATION.md) — Auditoria de validação científica por módulo
- [ROADMAP](docs/ROADMAP_commercial.md) — Roadmap estratégico
- [Tutoriais](tutoriais/) — Passo a passo

