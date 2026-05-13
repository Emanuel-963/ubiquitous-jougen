"""Gera Resumo28042026.pdf — relatório executivo do IonFlow Pipeline v0.3.0."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Resumo28042026.pdf"

# Windows system fonts (Unicode TrueType — suportam todos os caracteres PT)
_FONTS_DIR = r"C:\Windows\Fonts"
FONT_REG = os.path.join(_FONTS_DIR, "arial.ttf")
FONT_BOLD = os.path.join(_FONTS_DIR, "arialbd.ttf")
FONT_ITAL = os.path.join(_FONTS_DIR, "ariali.ttf")

# ── Paleta ────────────────────────────────────────────────────────────
AZUL = (26, 82, 118)  # título principal
AZUL_MED = (31, 119, 180)  # cabeçalhos de secção
VERDE = (39, 174, 96)  # checkmarks / positivo
CINZA_BG = (245, 248, 250)  # fundo de caixas
CINZA_TX = (80, 80, 80)  # texto secundário
PRETO = (30, 30, 30)  # corpo
BRANCO = (255, 255, 255)


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Arial", fname=FONT_REG)
        self.add_font("Arial", style="B", fname=FONT_BOLD)
        self.add_font("Arial", style="I", fname=FONT_ITAL)
        self.set_auto_page_break(auto=True, margin=18)

    def header(self):
        # Barra azul no topo
        self.set_fill_color(*AZUL)
        self.rect(0, 0, 210, 8, "F")
        # Linha fina dourada
        self.set_fill_color(212, 175, 55)
        self.rect(0, 8, 210, 1, "F")

    def footer(self):
        self.set_y(-13)
        self.set_font("Arial", "I", 8)
        self.set_text_color(*CINZA_TX)
        self.cell(
            0,
            6,
            f"IonFlow Pipeline v0.3.0  •  {date.today().strftime('%d/%m/%Y')}  •  Pág. {self.page_no()}",
            align="C",
        )
        self.set_fill_color(*AZUL)
        self.rect(0, 295, 210, 5, "F")

    # ── helpers ──────────────────────────────────────────────────────

    def section_title(self, text: str, top_margin: float = 6):
        self.ln(top_margin)
        # Barra colorida à esquerda
        self.set_fill_color(*AZUL_MED)
        self.rect(self.l_margin, self.get_y(), 3, 8, "F")
        self.set_x(self.l_margin + 5)
        self.set_font("Arial", "B", 13)
        self.set_text_color(*AZUL)
        self.cell(0, 8, text, ln=True)
        # Linha divisória
        self.set_draw_color(*AZUL_MED)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), 210 - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(*PRETO)

    def sub_title(self, text: str):
        self.ln(3)
        self.set_font("Arial", "B", 11)
        self.set_text_color(*AZUL_MED)
        self.cell(0, 7, text, ln=True)
        self.set_text_color(*PRETO)

    def body(self, text: str, indent: float = 0):
        self.set_font("Arial", "", 10)
        self.set_text_color(*PRETO)
        if indent:
            self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text: str, color=VERDE, symbol: str = ">"):
        self.set_x(self.l_margin)
        self.set_font("Arial", "B", 10)
        self.set_text_color(*color)
        self.cell(8, 6, symbol)
        self.set_font("Arial", "", 10)
        self.set_text_color(*PRETO)
        self.set_x(self.l_margin + 8)
        eff_w = self.w - self.l_margin - self.r_margin - 8
        self.multi_cell(eff_w, 6, text)

    def info_box(self, lines: list[str], fill=CINZA_BG):
        self.ln(2)
        self.set_fill_color(*fill)
        start_y = self.get_y()
        self.set_x(self.l_margin)
        # Medir altura
        h = len(lines) * 6 + 6
        self.rect(self.l_margin, start_y, 210 - self.l_margin - self.r_margin, h, "F")
        self.set_y(start_y + 3)
        for line in lines:
            self.set_font("Arial", "", 9)
            self.set_text_color(50, 50, 50)
            self.set_x(self.l_margin + 4)
            self.cell(0, 6, line, ln=True)
        self.set_text_color(*PRETO)
        self.ln(3)

    def metric_row(self, items: list[tuple]):
        """items: list of (label, value)."""
        w = (210 - self.l_margin - self.r_margin) / len(items)
        for label, value in items:
            x = self.get_x()
            y = self.get_y()
            self.set_fill_color(*AZUL)
            self.rect(x, y, w - 2, 14, "F")
            self.set_x(x)
            self.set_font("Arial", "B", 14)
            self.set_text_color(*BRANCO)
            self.cell(w - 2, 8, value, align="C")
            self.set_xy(x, y + 8)
            self.set_font("Arial", "", 7.5)
            self.cell(w - 2, 6, label, align="C")
            self.set_xy(x + w, y)
        self.ln(16)
        self.set_text_color(*PRETO)

    def two_col(self, left: list[str], right: list[str]):
        """Dois blocos side-by-side."""
        w = (210 - self.l_margin - self.r_margin - 4) / 2
        y_start = self.get_y()
        # Coluna esquerda
        for line in left:
            self.set_x(self.l_margin)
            self.set_font("Arial", "", 10)
            self.set_text_color(*PRETO)
            self.cell(w, 6, line, ln=True)
        y_end_l = self.get_y()
        # Coluna direita
        self.set_y(y_start)
        for line in right:
            self.set_x(self.l_margin + w + 4)
            self.set_font("Arial", "", 10)
            self.set_text_color(*PRETO)
            self.cell(w, 6, line, ln=True)
        self.set_y(max(y_end_l, self.get_y()))
        self.ln(2)


# ═══════════════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════════════

pdf = PDF()
pdf.set_margins(18, 14, 18)

# ── PÁGINA 1 — CAPA ──────────────────────────────────────────────────
pdf.add_page()
pdf.ln(12)

# Logo / título
pdf.set_fill_color(*AZUL)
pdf.rect(18, 22, 174, 38, "F")
pdf.set_y(28)
pdf.set_font("Arial", "B", 26)
pdf.set_text_color(*BRANCO)
pdf.cell(0, 12, "IonFlow Pipeline", align="C", ln=True)
pdf.set_font("Arial", "", 13)
pdf.cell(0, 9, "Plataforma de Análise Eletroquímica — v0.3.0", align="C", ln=True)
pdf.set_font("Arial", "I", 10)
pdf.cell(0, 7, "Relatório Executivo  \u2022  28 de Abril de 2026", align="C", ln=True)

pdf.ln(10)
pdf.set_text_color(*PRETO)
pdf.set_font("Arial", "", 11)
pdf.multi_cell(
    0,
    6.5,
    "O IonFlow Pipeline é um software open-source de análise de Espectroscopia "
    "de Impedância Eletroquímica (EIS), Ciclagem Galvanostática e Distribution "
    "of Relaxation Times (DRT), desenvolvido para investigadores em eletroquímica "
    "e ciência de materiais.",
)
pdf.ln(4)

# Métricas chave
pdf.section_title("Indicadores do Projeto")
pdf.metric_row(
    [
        ("Linhas de código", "~18 000"),
        ("Testes automatizados", "220+"),
        ("Fases implementadas", "7 / 7"),
        ("Versão", "v0.3.0"),
    ]
)

pdf.section_title("O Que o Software Faz")
bullets_o = [
    "Importa ficheiros de 4 fabricantes: Gamry, BioLogic, Autolab, Zahner",
    "Ajusta circuitos equivalentes automaticamente (Randles, CPE, Two-Arc...)",
    "Calcula parâmetros físicos: Rs, Rp, capacitância, energia, potência",
    "Executa análise DRT — distribui processos por constante de tempo τ",
    "Analisa ciclagem: eficiência, retenção de capacidade, diagrama de Ragone",
    "Gera relatórios PDF com gráficos e interpretação por IA (GPT/Ollama)",
    "Compara múltiplas amostras com Health Score e PCA automático",
    "Exporta para ZView, LaTeX booktabs, Origin, MEISP — pronto para artigos",
    "Guarda todos os resultados numa base de dados SQLite local rastreável",
    "Dashboard web Streamlit acessível pelo browser (localhost:8501)",
]
for b in bullets_o:
    pdf.bullet(b)

# ── PÁGINA 2 — ARQUITETURA ────────────────────────────────────────────
pdf.add_page()

pdf.section_title("Arquitetura do Sistema")
pdf.body(
    "O projeto segue uma arquitetura em camadas com separação clara entre "
    "parsers, lógica de análise, persistência e interface:"
)

layers = [
    ("Interface (GUI / CLI / Dashboard)", "gui_app.py  •  src/cli.py  •  dashboard.py"),
    (
        "Análise EIS / DRT / Ciclagem",
        "src/circuit_fitting.py  •  src/drt_analysis.py  •  src/cycling_calculator.py",
    ),
    (
        "Comparação & ML",
        "src/comparison/  •  src/ml_circuit_selector.py  •  src/pca_analysis.py",
    ),
    (
        "Parsers de Dados",
        "src/parsers/  (Gamry · BioLogic · Autolab · Zahner · genérico)",
    ),
    (
        "Exportação Científica",
        "src/export/  (ZView · LaTeX · Origin · MEISP · CSV · Excel · PDF)",
    ),
    ("IA Generativa", "src/ai/  (OpenAI · Ollama · LLM adapter)"),
    (
        "Base de Dados SQLite",
        "src/db/  (IonFlowRepository · FeatureStoreV2 · migrations)",
    ),
    ("Configuração & Update", "src/config.py  •  src/updater.py"),
]

for layer, detail in layers:
    pdf.set_fill_color(230, 240, 255)
    x, y = pdf.get_x(), pdf.get_y()
    w = 210 - pdf.l_margin - pdf.r_margin
    pdf.rect(x, y, w, 10, "F")
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(*AZUL)
    pdf.set_x(x + 3)
    pdf.cell(70, 10, layer)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(*PRETO)
    pdf.cell(0, 10, detail, ln=True)
    pdf.ln(1)

pdf.ln(4)
pdf.section_title("Tecnologia Utilizada")
pdf.two_col(
    [
        "Linguagem:      Python 3.10+",
        "GUI:            CustomTkinter 5.2.2",
        "Cálculo:        NumPy · SciPy · scikit-learn",
        "Gráficos:       Matplotlib · mplcursors",
        "Dashboard web:  Streamlit ≥1.30",
    ],
    [
        "Base de dados:  SQLite3 (stdlib)",
        "Exportação:     fpdf2 · openpyxl",
        "IA / LLM:       OpenAI API · Ollama",
        "Testes:         pytest · pytest-cov",
        "CI:             pre-commit (black · isort · flake8)",
    ],
)

# ── PÁGINA 3 — FASES v0.3.0 ──────────────────────────────────────────
pdf.add_page()
pdf.section_title("Fases Implementadas na v0.3.0")

fases = [
    (
        "Fase 1",
        "Quick Wins",
        "Relatório LaTeX com tabelas booktabs; pre-commit hooks (black/isort/flake8); "
        "base de código limpa e testada.",
    ),
    (
        "Fase 2",
        "Importação Direta de Potenciostatos",
        "Parsers nativos para Gamry (.dta), BioLogic (.mpr/.mpt), Metrohm Autolab (.csv) "
        "e Zahner (.isc). API unificada parse_eis_file() com deteção automática de formato.",
    ),
    (
        "Fase 3",
        "Exportação Científica",
        "Quatro novos formatos: ZView/ZPlot (.z), LaTeX booktabs (.tex), "
        "OriginPro (.csv com metadados) e MEISP (.txt). Prontos para submissão em revista.",
    ),
    (
        "Fase 4",
        "Análise Comparativa",
        "Aba '📊 Comparar' na GUI com sobreposição de Nyquist/Bode, Health Score (0–10), "
        "PCA automático com clustering e tabela de ranking exportável.",
    ),
    (
        "Fase 5",
        "IA Generativa",
        "Integração LLM com OpenAI (gpt-4o, gpt-4o-mini) e Ollama local (llama3, mistral). "
        "Narrativa científica automática incluída nos relatórios PDF.",
    ),
    (
        "Fase 6",
        "Painel de Configurações + Auto-Update",
        "Painel '⚙️ Configurações' persistente em JSON. Verificação automática de "
        "novas versões no GitHub com download integrado na GUI.",
    ),
    (
        "Fase 7",
        "SQLite Backend + Dashboard Streamlit",
        "Base de dados local com 7 tabelas (amostras, EIS, DRT, ciclagem, parâmetros, "
        "histórico ML). Dashboard web com 7 páginas acessível em localhost:8501.",
    ),
]

for tag, titulo, desc in fases:
    # Caixa de fase
    y0 = pdf.get_y()
    pdf.set_fill_color(*AZUL)
    pdf.rect(18, y0, 18, 11, "F")
    pdf.set_xy(18, y0)
    pdf.set_font("Arial", "B", 7.5)
    pdf.set_text_color(*BRANCO)
    pdf.cell(18, 11, tag, align="C")
    # Título da fase
    pdf.set_x(38)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(*AZUL)
    pdf.cell(0, 6, titulo, ln=True)
    # Descrição
    pdf.set_x(38)
    pdf.set_font("Arial", "", 9.5)
    pdf.set_text_color(*PRETO)
    pdf.multi_cell(0, 5.5, desc)
    # Checkmark verde
    pdf.set_xy(190, y0)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(*VERDE)
    pdf.cell(0, 11, "[OK] Concluido")
    pdf.ln(3)

# ── PÁGINA 4 — COMO USAR ─────────────────────────────────────────────
pdf.add_page()
pdf.section_title("Como Instalar e Usar")

pdf.sub_title("Instalação (uma única vez)")
pdf.info_box(
    [
        "git clone https://github.com/Emanuel-963/ubiquitous-jougen.git",
        "cd eis_analytics",
        "python -m venv venv",
        ".\\venv\\Scripts\\Activate.ps1          # Windows PowerShell",
        "pip install -e .[dashboard]             # inclui Streamlit",
    ]
)

pdf.sub_title("Lançar a Interface Gráfica (GUI)")
pdf.info_box(
    [
        "python gui_app.py",
    ]
)

pdf.sub_title("Lançar o Dashboard Web")
pdf.info_box(
    [
        ".\\venv\\Scripts\\python.exe -m streamlit run dashboard.py",
        "# Abre automaticamente em http://localhost:8501",
    ]
)

pdf.sub_title("Linha de Comandos (CLI)")
pdf.info_box(
    [
        "ionflow-cli eis      --data-dir data/raw/ --output outputs/",
        "ionflow-cli cycling  --data-dir data/raw/ --output outputs/",
        "ionflow-cli drt      --data-dir data/raw/ --output outputs/",
        "ionflow-cli analyze  --all --ai --export-pdf relatorio.pdf",
    ]
)

pdf.sub_title("Usar via Python")
pdf.info_box(
    [
        "from src.parsers import parse_eis_file",
        "from src.db import IonFlowRepository",
        "",
        "df = parse_eis_file('medição.dta').data",
        "repo = IonFlowRepository('data/ionflow.db')",
        "sid = repo.add_sample('AM1', 'eis')",
        "repo.save_eis_results(sid, df_params)",
    ]
)

pdf.section_title("Fluxo de Trabalho Típico")
steps = [
    (
        "1",
        "Colocar ficheiros EIS na pasta data/raw/",
        "Formatos: .dta, .mpr, .mpt, .csv, .txt, .xlsx",
    ),
    (
        "2",
        "Executar o pipeline EIS na GUI ou CLI",
        "Todos os resultados ficam em outputs/ e na BD SQLite",
    ),
    (
        "3",
        "Abrir o Dashboard para visualizar e explorar",
        "python -m streamlit run dashboard.py",
    ),
    (
        "4",
        "Comparar amostras na aba Comparar da GUI",
        "Health Score + PCA + ranking automático",
    ),
    (
        "5",
        "Pedir interpretação ao agente IA",
        "Selecionar provedor LLM e clicar 'Analisar com IA'",
    ),
    (
        "6",
        "Exportar tabela LaTeX + gráficos PNG para o artigo",
        "Formatos: .tex, .z, .csv Origin, imagens 300 DPI",
    ),
]
for num, action, detail in steps:
    y0 = pdf.get_y()
    pdf.set_fill_color(*AZUL_MED)
    pdf.rect(18, y0, 8, 10, "F")
    pdf.set_xy(18, y0)
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(*BRANCO)
    pdf.cell(8, 10, num, align="C")
    pdf.set_x(28)
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(*PRETO)
    pdf.cell(0, 6, action, ln=True)
    pdf.set_x(28)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(*CINZA_TX)
    pdf.cell(0, 5, detail, ln=True)
    pdf.ln(1)

# ── PÁGINA 5 — FUNCIONALIDADES / REPOSITÓRIO ─────────────────────────
pdf.add_page()
pdf.section_title("Funcionalidades por Módulo")

modulos = [
    (
        "Parsers (src/parsers/)",
        [
            "Auto-deteção de formato por magic bytes + extensão",
            "Gamry .dta, BioLogic .mpr/.mpt, Autolab, Zahner .isc",
            "Fallback genérico CSV / TXT / Excel",
            "Retorna ParsedEIS com .data (DataFrame) e .metadata (dict)",
        ],
    ),
    (
        "Circuit Fitting (src/circuit_fitting.py)",
        [
            "10+ circuitos pré-definidos: Randles, Two-Arc, CPE, Indutivo...",
            "Auto-composição de circuitos combinatórios",
            "Scoring multi-critério: BIC, confiança ML, validação KK",
            "Ranking automático com categorias em português",
        ],
    ),
    (
        "DRT (src/drt_analysis.py)",
        [
            "Tikhonov regularization no domínio do tempo de relaxação",
            "Extração automática de picos τ e γ",
            "Visualização stem-plot + distribuição suavizada",
        ],
    ),
    (
        "Ciclagem (src/cycling_calculator.py + loader.py)",
        [
            "Cálculo de energia (Wh/kg) e potência (W/kg)",
            "Eficiência coulômbica ciclo a ciclo",
            "Retenção de capacidade (critério 80%)",
            "Diagrama de Ragone interativo",
        ],
    ),
    (
        "Comparação (src/comparison/)",
        [
            "Sobreposição N amostras: Nyquist + Bode + DRT",
            "Health Score ponderado (Rs, Rp, BIC, KK, confiança)",
            "PCA com clustering k-means automático",
            "Relatório comparativo PDF",
        ],
    ),
    (
        "IA / LLM (src/ai/)",
        [
            "LLMAdapter abstrato — OpenAI + Ollama + NullAdapter",
            "Narrativa científica em português gerada automaticamente",
            "Enriquecimento do relatório PDF com interpretação",
            "Dashboard: página dedicada para análise interativa",
        ],
    ),
    (
        "Base de Dados (src/db/)",
        [
            "7 tabelas SQLite com FK CASCADE e índices",
            "IonFlowRepository: CRUD completo (samples, EIS, DRT, cycling, params)",
            "FeatureStoreV2: histórico ML com spectral features e similarity search",
            "Migrações de schema versionadas e idempotentes",
        ],
    ),
    (
        "Dashboard (dashboard.py)",
        [
            "7 páginas Streamlit: Visão Geral, Upload, EIS, DRT, Ciclagem, Histórico, IA",
            "Acesso local (localhost:8501) ou em rede local para apresentações",
            "Upload de ficheiros EIS com pré-visualização Nyquist",
            "Análise IA interativa no browser",
        ],
    ),
]

for mod, items in modulos:
    pdf.sub_title(mod)
    for item in items:
        pdf.bullet(item, color=AZUL_MED, symbol="->")
    pdf.ln(1)

# ── PÁGINA 6 — QUALIDADE / REPOSITÓRIO / CONTACTO ─────────────────────
pdf.add_page()
pdf.section_title("Qualidade e Testes")

pdf.body(
    "O projeto tem uma suite completa de testes automatizados cobrindo todos "
    "os módulos principais. Os testes correm a cada commit via pre-commit hooks."
)

qualidade = [
    ("220+ testes automatizados", "pytest · pytest-cov"),
    ("Cobertura de código medida", "htmlcov/ — relatório HTML gerado localmente"),
    ("Linting automático", "flake8 7.1.1 — sem erros em todos os módulos"),
    ("Formatação consistente", "black 23.12.0 + isort 5.12.0"),
    (
        "Sem dependências de segurança (OWASP)",
        "Entradas validadas em todas as fronteiras",
    ),
    ("Compatibilidade Python", "3.10, 3.11, 3.12, 3.14 (testado)"),
]

for feat, detalhe in qualidade:
    y0 = pdf.get_y()
    pdf.bullet(f"{feat}  —  {detalhe}", color=VERDE)

pdf.section_title("Repositório e Acesso")

pdf.set_fill_color(240, 248, 255)
pdf.rect(18, pdf.get_y(), 174, 28, "F")
pdf.ln(3)
pdf.set_font("Arial", "B", 10)
pdf.set_text_color(*AZUL)
pdf.cell(0, 6, "GitHub:", ln=True)
pdf.set_font("Arial", "", 10)
pdf.set_text_color(*PRETO)
pdf.cell(0, 6, "  https://github.com/Emanuel-963/ubiquitous-jougen", ln=True)
pdf.set_font("Arial", "B", 10)
pdf.set_text_color(*AZUL)
pdf.cell(0, 6, "Branch / versão atual:", ln=True)
pdf.set_font("Arial", "", 10)
pdf.set_text_color(*PRETO)
pdf.cell(0, 6, "  main  •  commit fb0badb  •  v0.3.0  •  28 Abril 2026", ln=True)
pdf.ln(5)

pdf.section_title("Ficheiros de Documentação Disponíveis")

docs = [
    ("CHANGELOG.md", "Histórico completo de versões e commits"),
    ("README.md", "Guia de instalação e início rápido"),
    ("docs/ROADMAP_v0.3.0.md", "Plano detalhado das 7 fases de desenvolvimento"),
    ("docs/RECOMMENDATIONS_*.md", "Recomendações técnicas pós-v0.2.0"),
    ("tutoriais/01 … 22", "22 tutoriais em português cobrindo todas as funções"),
    ("htmlcov/index.html", "Relatório de cobertura de testes (abrir no browser)"),
]

for ficheiro, desc in docs:
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(*AZUL_MED)
    pdf.cell(70, 6, ficheiro)
    pdf.set_font("Arial", "", 9.5)
    pdf.set_text_color(*PRETO)
    pdf.cell(0, 6, desc, ln=True)

pdf.ln(6)
pdf.section_title("Proximos Passos Sugeridos")
proximos = [
    "Publicar no PyPI para instalação via pip install ionflow-pipeline",
    "Criar instalador .exe Windows com PyInstaller (IonFlow_Pipeline_Setup.exe)",
    "Adicionar parser Solartron / ZAHNER Z (extensão da Fase 2)",
    "Integrar validação estatística de Bootstrap para incertezas de fitting",
    "Containerizar com Docker para execução em servidores de laboratório",
    "Submeter artigo de software (Journal of Open Source Software — JOSS)",
]
for p in proximos:
    pdf.bullet(p, color=(180, 140, 20), symbol=">>")

# ── Rodapé da última página ──────────────────────────────────────────
pdf.ln(8)
pdf.set_fill_color(*AZUL)
pdf.rect(18, pdf.get_y(), 174, 1, "F")
pdf.ln(3)
pdf.set_font("Arial", "I", 9)
pdf.set_text_color(*CINZA_TX)
pdf.multi_cell(
    0,
    5.5,
    "Este documento foi gerado automaticamente pelo IonFlow Pipeline em "
    f"{date.today().strftime('%d de %B de %Y')}. "
    "Para aceder ao código fonte completo e documentação técnica, visite o repositório GitHub acima.",
)

# ── SALVAR ────────────────────────────────────────────────────────────
pdf.output(str(OUT))
print(f"PDF gerado: {OUT}")
print(f"Tamanho: {OUT.stat().st_size / 1024:.0f} KB")
