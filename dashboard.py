"""IonFlow Pipeline — Streamlit Dashboard (Phase 7).

Run locally
-----------
::

    streamlit run dashboard.py

Or with the optional Docker image::

    docker compose up dashboard

Install the optional dependency first::

    pip install ionflow-pipeline[dashboard]
    # or: pip install streamlit>=1.30
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running from any cwd
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402 — must set backend before pyplot import

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import __version__  # noqa: E402
from src.dashboard import nyquist_fig, parse_uploaded_file  # noqa: E402
from src.db.feature_store_v2 import FeatureStoreV2  # noqa: E402
from src.db.repository import IonFlowRepository  # noqa: E402

# ── Page config (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="IonFlow Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_DB_PATH = os.environ.get("IONFLOW_DB", str(_ROOT / "data" / "ionflow.db"))


# ── Cached resources ──────────────────────────────────────────────────


@st.cache_resource
def _get_repo() -> IonFlowRepository:
    return IonFlowRepository(_DB_PATH)


@st.cache_resource
def _get_feature_store() -> FeatureStoreV2:
    return FeatureStoreV2(_DB_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────

PAGES = {
    "🏠 Visão Geral": "overview",
    "📤 Upload & Dados": "upload",
    "📊 Resultados EIS": "eis",
    "🌀 Análise DRT": "drt",
    "🔋 Ciclagem": "cycling",
    "🗄️ Histórico ML": "history",
    "🤖 Análise IA": "ai",
}

with st.sidebar:
    st.markdown("## ⚡ IonFlow")
    st.caption(f"Pipeline v{__version__}  |  Dashboard v0.3.0")
    st.divider()
    selected_label = st.radio(
        "Página",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"DB: `{Path(_DB_PATH).name}`")

_PAGE_ID = PAGES[selected_label]


# ═══════════════════════════════════════════════════════════════════════
# Page: Visão Geral
# ═══════════════════════════════════════════════════════════════════════


def _page_overview() -> None:
    st.title("🏠 Visão Geral")
    st.markdown(f"**IonFlow Pipeline** `v{__version__}` — SQLite Dashboard")

    repo = _get_repo()
    stats = repo.stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Amostras", stats.get("samples", 0))
    col2.metric("Resultados EIS", stats.get("eis_results", 0))
    col3.metric("Ciclagem", stats.get("cycling_results", 0))
    col4.metric("Histórico ML", stats.get("fitting_history", 0))

    st.divider()

    samples_df = repo.get_all_samples()
    if not samples_df.empty:
        st.subheader("Amostras recentes")
        st.dataframe(
            samples_df.head(20),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Nenhuma amostra armazenada ainda. "
            "Use **📤 Upload & Dados** para importar ficheiros EIS."
        )

    st.divider()
    st.markdown(
        """
### Guia rápido

| Página | Função |
|--------|--------|
| **📤 Upload & Dados** | Carregar ficheiros EIS; pré-visualização e gravação no BD |
| **📊 Resultados EIS** | Tabelas de parâmetros + gráficos Nyquist / Bode |
| **🌀 Análise DRT** | Picos τ / γ por amostra |
| **🔋 Ciclagem** | Tabela de ciclos + diagrama de Ragone |
| **🗄️ Histórico ML** | Navegador do FeatureStore SQLite |
| **🤖 Análise IA** | Narração LLM dos resultados |

Inicie o IonFlow GUI para executar o pipeline completo de análise,
depois os resultados serão guardados aqui automaticamente.
        """
    )


# ═══════════════════════════════════════════════════════════════════════
# Page: Upload & Dados
# ═══════════════════════════════════════════════════════════════════════


def _page_upload() -> None:
    st.title("📤 Upload & Dados")
    st.markdown(
        "Carregue um ou mais ficheiros EIS para pré-visualização e gravação no BD."
    )

    uploaded = st.file_uploader(
        "Selecionar ficheiros EIS",
        type=["csv", "txt", "dta", "mpr", "mpt", "xlsx"],
        accept_multiple_files=True,
        help="Formatos aceites: CSV/TXT genérico, Gamry .dta, BioLogic .mpr/.mpt, Excel .xlsx",
    )

    if not uploaded:
        st.info("Nenhum ficheiro seleccionado.")
        return

    parsed: list = []
    for f in uploaded:
        df = parse_uploaded_file(f)
        if df is None:
            st.warning(f"⚠️ Não foi possível ler **{f.name}** como EIS.")
        else:
            parsed.append((f.name, df))

    if not parsed:
        return

    st.success(f"{len(parsed)} ficheiro(s) lido(s) com sucesso.")

    tabs = st.tabs([name for name, _ in parsed])
    for tab, (name, df) in zip(tabs, parsed):
        with tab:
            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown(f"**{name}** — {len(df)} pontos")
                st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            with col_r:
                fig = nyquist_fig(df, title=name)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    st.divider()
    run_label = st.text_input(
        "Rótulo da corrida",
        value=pd.Timestamp.now().strftime("Run %Y-%m-%d %H:%M"),
    )

    if st.button("💾 Guardar no BD", type="primary"):
        repo = _get_repo()
        saved = 0
        for name, df in parsed:
            sample_id = repo.add_sample(name, "eis", file_path=name)
            # Store frequency/zreal/zimag as a minimal EIS result row
            mini = pd.DataFrame(
                [
                    {
                        "file_label": name,
                        "n_points": len(df),
                        "freq_min": float(df["frequency"].min()),
                        "freq_max": float(df["frequency"].max()),
                        "zreal_min": float(df["zreal"].min()),
                        "zreal_max": float(df["zreal"].max()),
                    }
                ],
                index=[name],
            )
            repo.save_eis_results(sample_id, mini)
            saved += 1
        # Invalidate cached resource so stats refresh
        _get_repo.clear()
        st.success(f"✅ {saved} amostra(s) guardada(s) com rótulo «{run_label}».")


# ═══════════════════════════════════════════════════════════════════════
# Page: Resultados EIS
# ═══════════════════════════════════════════════════════════════════════


def _page_eis() -> None:
    st.title("📊 Resultados EIS")

    repo = _get_repo()
    df = repo.get_eis_results()

    if df.empty:
        st.info("Nenhum resultado EIS no BD. Execute o pipeline ou carregue ficheiros.")
        return

    # ── Filters ──────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    samples = ["Todos"] + sorted(df["sample_name"].dropna().unique().tolist())
    sel_sample = col_f1.selectbox("Amostra", samples)
    circuits = ["Todos"] + sorted(df["circuit_name"].dropna().unique().tolist())
    sel_circuit = col_f2.selectbox("Circuito", circuits)

    filtered = df.copy()
    if sel_sample != "Todos":
        filtered = filtered[filtered["sample_name"] == sel_sample]
    if sel_circuit != "Todos":
        filtered = filtered[filtered["circuit_name"] == sel_circuit]

    display_cols = [
        c
        for c in [
            "sample_name",
            "file_label",
            "circuit_name",
            "rs_fit",
            "rp_fit",
            "c_mean",
            "energy_mean",
            "score",
            "rank",
            "category",
        ]
        if c in filtered.columns
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    st.caption(f"{len(filtered)} resultados")


# ═══════════════════════════════════════════════════════════════════════
# Page: Análise DRT
# ═══════════════════════════════════════════════════════════════════════


def _page_drt() -> None:
    st.title("🌀 Análise DRT")

    repo = _get_repo()
    df = repo.get_drt_results()

    if df.empty:
        st.info(
            "Nenhum resultado DRT no BD. Execute a análise DRT via GUI e guarde no BD."
        )
        return

    samples = ["Todos"] + sorted(df["sample_name"].dropna().unique().tolist())
    sel = st.selectbox("Amostra", samples)
    if sel != "Todos":
        df = df[df["sample_name"] == sel]

    display_cols = [
        c
        for c in [
            "sample_name",
            "file_label",
            "tau_peak1",
            "gamma_peak1",
            "tau_peak2",
            "gamma_peak2",
            "tau_peak3",
            "gamma_peak3",
        ]
        if c in df.columns
    ]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    # Stem-plot of peaks for the selected sample
    peak_rows = df.dropna(subset=["tau_peak1", "gamma_peak1"])
    if not peak_rows.empty:
        st.subheader("Picos DRT")
        fig, ax = plt.subplots(figsize=(7, 4))
        for _, row in peak_rows.iterrows():
            for i in range(1, 4):
                tau = row.get(f"tau_peak{i}")
                gamma = row.get(f"gamma_peak{i}")
                if tau and gamma:
                    ax.vlines(
                        np.log10(float(tau)),
                        0,
                        float(gamma),
                        colors="#1b4f72",
                        linewidth=2,
                    )
                    ax.scatter(
                        np.log10(float(tau)), float(gamma), color="#1b4f72", zorder=5
                    )
        ax.set_xlabel("log₁₀(τ / s)")
        ax.set_ylabel("γ(τ)")
        ax.set_title("Picos DRT")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Page: Ciclagem
# ═══════════════════════════════════════════════════════════════════════


def _page_cycling() -> None:
    st.title("🔋 Ciclagem")

    repo = _get_repo()
    df = repo.get_cycling_results()

    if df.empty:
        st.info("Nenhum resultado de ciclagem no BD.")
        return

    samples = ["Todos"] + sorted(df["sample_name"].dropna().unique().tolist())
    sel = st.selectbox("Amostra", samples)
    if sel != "Todos":
        df = df[df["sample_name"] == sel]

    display_cols = [
        c
        for c in [
            "sample_name",
            "cycle_number",
            "energy_wh_kg",
            "power_w_kg",
            "retention_pct",
        ]
        if c in df.columns
    ]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    # Ragone plot
    rag = df.dropna(subset=["energy_wh_kg", "power_w_kg"])
    if not rag.empty:
        st.subheader("Diagrama de Ragone")
        fig, ax = plt.subplots(figsize=(6, 5))
        if "sample_name" in rag.columns:
            for name, grp in rag.groupby("sample_name"):
                ax.scatter(
                    grp["power_w_kg"], grp["energy_wh_kg"], label=name, alpha=0.7, s=40
                )
            ax.legend(fontsize=8)
        else:
            ax.scatter(
                rag["power_w_kg"], rag["energy_wh_kg"], color="#1b4f72", s=40, alpha=0.7
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Potência (W/kg)")
        ax.set_ylabel("Energia (Wh/kg)")
        ax.set_title("Ragone")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Retention curve
    ret = df.dropna(subset=["cycle_number", "retention_pct"])
    if not ret.empty:
        st.subheader("Retenção de Capacidade")
        fig, ax = plt.subplots(figsize=(7, 4))
        if "sample_name" in ret.columns:
            for name, grp in ret.groupby("sample_name"):
                grp_s = grp.sort_values("cycle_number")
                ax.plot(grp_s["cycle_number"], grp_s["retention_pct"], label=name)
            ax.legend(fontsize=8)
        else:
            ret_s = ret.sort_values("cycle_number")
            ax.plot(ret_s["cycle_number"], ret_s["retention_pct"], color="#1b4f72")
        ax.set_xlabel("Ciclo")
        ax.set_ylabel("Retenção (%)")
        ax.set_title("Retenção de Capacidade")
        ax.axhline(80, color="red", linestyle="--", alpha=0.5, label="80 %")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Page: Histórico ML
# ═══════════════════════════════════════════════════════════════════════


def _page_history() -> None:
    st.title("🗄️ Histórico ML")
    st.markdown("Navegue no histórico de fitting guardado no FeatureStore SQLite.")

    store = _get_feature_store()
    total = len(store)

    if total == 0:
        st.info(
            "Histórico vazio. O histórico é populado automaticamente "
            "ao executar o pipeline com circuit fitting activado."
        )
        return

    st.metric("Total de registos", total)
    st.text(store.summary_text())

    st.divider()

    # ── Circuit stats ─────────────────────────────────────────────────
    stats = store.circuit_stats()
    if stats:
        st.subheader("Estatísticas por Circuito")
        stats_df = (
            pd.DataFrame(stats)
            .T.reset_index()
            .rename(columns={"index": "circuit_name"})
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        names = [r["circuit_name"] for r in stats_df.to_dict("records")]
        counts = [r["count"] for r in stats_df.to_dict("records")]
        ax.barh(names, counts, color="#1b4f72")
        ax.set_xlabel("Frequência de uso")
        ax.set_title("Circuitos mais usados")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()

    # ── Record browser ────────────────────────────────────────────────
    st.subheader("Registos")
    circuits = ["Todos"] + store.unique_circuits()
    sel_circ = st.selectbox("Filtrar por circuito", circuits)

    records = store.query(circuit_name=sel_circ if sel_circ != "Todos" else None)
    if records:
        preview = [
            {
                "sample_id": r.get("sample_id"),
                "circuit_name": r.get("circuit_name"),
                "bic": r.get("bic"),
                "confidence": r.get("confidence"),
                "created_at": r.get("created_at"),
            }
            for r in records[:200]
        ]
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
    else:
        st.info("Sem registos para o filtro seleccionado.")

    st.divider()

    if st.button("🗑️ Limpar histórico", type="secondary"):
        store.clear()
        _get_feature_store.clear()
        st.success("Histórico limpo.")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# Page: Análise IA
# ═══════════════════════════════════════════════════════════════════════


def _page_ai() -> None:
    st.title("🤖 Análise IA")
    st.markdown(
        "Gere uma narrativa científica dos seus resultados usando um LLM local "
        "(Ollama) ou na nuvem (OpenAI)."
    )

    col_cfg, col_run = st.columns([1, 2])

    with col_cfg:
        provider = st.selectbox(
            "Provedor",
            ["none", "ollama", "openai"],
            help="'none' desactiva o LLM",
        )
        model = st.text_input(
            "Modelo", value="llama3" if provider == "ollama" else "gpt-4o-mini"
        )
        api_key = ""
        base_url = ""
        if provider == "openai":
            api_key = st.text_input("API Key", type="password")
            base_url = st.text_input("Base URL (opcional)", value="")
        elif provider == "ollama":
            base_url = st.text_input("Base URL", value="http://localhost:11434")
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.3, 0.05)

    with col_run:
        context = st.text_area(
            "Dados / contexto para análise",
            height=200,
            placeholder=(
                "Cole aqui os resultados EIS (Rs, Rp, circuito...) "
                "ou DRT (picos τ/γ) para obter uma narrativa automática."
            ),
        )
        prompt = st.text_input(
            "Instrução adicional",
            value="Analise os dados e forneça uma interpretação científica em português.",
        )

        if st.button("▶ Executar", type="primary", disabled=(provider == "none")):
            if not context.strip():
                st.warning("Por favor insira dados no campo 'contexto'.")
                return
            try:
                from src.ai.llm_adapter import LLMConfig, LLMProvider, create_adapter

                _provider_map = {
                    "openai": LLMProvider.OPENAI,
                    "ollama": LLMProvider.OLLAMA,
                }
                cfg = LLMConfig(
                    provider=_provider_map.get(provider, LLMProvider.NONE),
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                )
                adapter = create_adapter(cfg)
                with st.spinner("A processar…"):
                    response = adapter.chat(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Você é um especialista em electroquímica "
                                    "que analisa dados de EIS (Espectroscopia de "
                                    "Impedância Electroquímica)."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"{prompt}\n\nDados:\n{context}",
                            },
                        ]
                    )
                st.markdown("### Resultado")
                st.markdown(response)
            except Exception as exc:
                st.error(f"Erro: {exc}")

        if provider == "none":
            st.info("Selecione um provedor LLM para activar a análise.")


# ═══════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════

_ROUTER = {
    "overview": _page_overview,
    "upload": _page_upload,
    "eis": _page_eis,
    "drt": _page_drt,
    "cycling": _page_cycling,
    "history": _page_history,
    "ai": _page_ai,
}

_ROUTER[_PAGE_ID]()
