"""
llm-bench — Local LLM Comparison Dashboard
=========================================
Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── ensure project root is on the path ──────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from llm_bench.models.registry import MODELS, get_families, get_all_tags, list_models_by_vram

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="llm-bench | Local LLM Comparison",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e94560;
  }
  .metric-label { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }

  .hw-badge {
    display: inline-block;
    background: #0f3460;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e94560;
    margin: 0.2rem;
  }

  .tag-pill {
    display: inline-block;
    background: rgba(233,69,96,0.15);
    border: 1px solid rgba(233,69,96,0.4);
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #e94560;
    margin: 1px;
  }

  .stButton>button {
    background: linear-gradient(90deg, #e94560, #c62a47);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(233,69,96,0.4); }

  .recommend-box {
    background: linear-gradient(135deg, #0d2137, #122b45);
    border-left: 4px solid #e94560;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_precomputed() -> dict:
    p = ROOT / "data" / "precomputed" / "all_results.json"
    with open(p) as f:
        return json.load(f)


@st.cache_data
def build_results_df(hw_tag: str, precomputed: dict) -> pd.DataFrame:
    speed_rows = precomputed["speed_results"].get(hw_tag, [])
    quality_mmlu = {
        r["model_id"]: r["score"]
        for r in precomputed["quality_results"].get("mmlu", [])
    }
    quality_he = {
        r["model_id"]: r["score"]
        for r in precomputed["quality_results"].get("humaneval", [])
    }

    rows = []
    for s in speed_rows:
        mid = s["model_id"]
        meta = MODELS.get(mid, {})
        rows.append({
            "model_id": mid,
            "name": meta.get("name", mid),
            "family": meta.get("family", "?"),
            "params": meta.get("params", "?"),
            "quantization": s["quantization"],
            "tokens_per_sec": s.get("tokens_per_sec", 0),
            "ttft_ms": s.get("ttft_ms", 0),
            "memory_gb": s.get("memory_gb", 0),
            "mmlu": quality_mmlu.get(mid, None),
            "humaneval": quality_he.get(mid, None),
            "context_k": meta.get("context", 4096) // 1024,
            "tags": ", ".join(meta.get("tags", [])),
        })

    return pd.DataFrame(rows)


@st.cache_data
def detect_hw_safe() -> dict:
    try:
        from llm_bench.utils.hardware_detect import detect_hardware
        hw = detect_hardware()
        return hw.to_dict()
    except Exception as e:
        return {"error": str(e), "has_gpu": False, "total_vram_gb": 0, "ram_total_gb": 16}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    st.sidebar.markdown("## 🦙 llm-bench")
    st.sidebar.markdown("*Compare local LLMs on your hardware*")
    st.sidebar.divider()

    # Hardware profile selector
    st.sidebar.markdown("### 🖥️ Hardware Profile")
    hw_options = {
        "RTX 3090 (24 GB)": "rtx3090",
        "RTX 4090 (24 GB)": "rtx4090",
        "Apple M2 Max (96 GB)": "m2_max",
        "NVIDIA A100 80 GB": "a100_80gb",
        "🔴 My Hardware (detect)": "live",
    }
    hw_label = st.sidebar.selectbox("Select hardware", list(hw_options.keys()))
    hw_tag = hw_options[hw_label]

    st.sidebar.divider()
    st.sidebar.markdown("### 🔬 Filters")

    families = ["All"] + sorted(get_families())
    family_filter = st.sidebar.selectbox("Model family", families)

    quant_filter = st.sidebar.multiselect(
        "Quantization",
        ["fp16", "8bit", "4bit"],
        default=["4bit"],
    )

    min_speed = st.sidebar.slider("Min speed (tok/s)", 0, 250, 0, step=5)
    min_mmlu = st.sidebar.slider("Min MMLU accuracy (%)", 0, 100, 0, step=5) / 100

    st.sidebar.divider()
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown(
        "Open-source benchmark tool. "
        "[⭐ Star on GitHub](https://github.com/your-org/llm-bench)"
    )

    return {
        "hw_tag": hw_tag,
        "family_filter": family_filter,
        "quant_filter": quant_filter,
        "min_speed": min_speed,
        "min_mmlu": min_mmlu,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Page sections
# ─────────────────────────────────────────────────────────────────────────────

def render_hardware_banner(hw_tag: str, precomputed: dict) -> None:
    hw_profiles = precomputed.get("hardware_profiles", {})
    hw = hw_profiles.get(hw_tag, {})
    if not hw:
        return

    st.markdown("#### 🖥️ Hardware Profile")
    cols = st.columns(4)
    cols[0].markdown(
        f'<div class="metric-card"><div class="metric-value">{hw.get("gpu","?").split()[-1]}</div>'
        f'<div class="metric-label">GPU</div></div>', unsafe_allow_html=True
    )
    cols[1].markdown(
        f'<div class="metric-card"><div class="metric-value">{hw.get("vram_total_gb",0):.0f} GB</div>'
        f'<div class="metric-label">VRAM</div></div>', unsafe_allow_html=True
    )
    cols[2].markdown(
        f'<div class="metric-card"><div class="metric-value">{hw.get("ram_total_gb",0):.0f} GB</div>'
        f'<div class="metric-label">System RAM</div></div>', unsafe_allow_html=True
    )
    cols[3].markdown(
        f'<div class="metric-card"><div class="metric-value">{hw.get("cpu_cores",0)}C</div>'
        f'<div class="metric-label">CPU cores</div></div>', unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)


def render_scatter(df: pd.DataFrame) -> None:
    st.markdown("### ⚡ Speed vs Quality")
    if df.empty or df["mmlu"].isna().all():
        st.info("No quality data available for this selection.")
        return

    plot_df = df.dropna(subset=["mmlu"])
    fig = px.scatter(
        plot_df,
        x="tokens_per_sec",
        y="mmlu",
        color="family",
        size="memory_gb",
        size_max=30,
        hover_name="name",
        hover_data={"params": True, "quantization": True, "ttft_ms": True, "memory_gb": ":.1f"},
        labels={
            "tokens_per_sec": "Speed (tokens / second)",
            "mmlu": "MMLU Accuracy",
        },
        title="Speed vs Quality — bubble size = VRAM usage",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        yaxis_tickformat=".0%",
        height=460,
    )
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    st.plotly_chart(fig, use_container_width=True)


def render_speed_bars(df: pd.DataFrame) -> None:
    st.markdown("### 🚀 Generation Speed (tokens/second)")
    sdf = df.sort_values("tokens_per_sec", ascending=True).tail(20)
    fig = px.bar(
        sdf,
        x="tokens_per_sec",
        y="name",
        orientation="h",
        color="family",
        hover_data=["quantization", "memory_gb", "ttft_ms"],
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"tokens_per_sec": "Tokens / second", "name": ""},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=max(350, len(sdf) * 28),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_memory_chart(df: pd.DataFrame) -> None:
    st.markdown("### 💾 VRAM Usage")
    mdf = df.sort_values("memory_gb")
    fig = px.bar(
        mdf,
        x="name",
        y="memory_gb",
        color="quantization",
        template="plotly_dark",
        barmode="group",
        labels={"memory_gb": "VRAM (GB)", "name": ""},
        color_discrete_map={"fp16": "#e94560", "8bit": "#f5a623", "4bit": "#7ed321"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_quality_radar(df: pd.DataFrame) -> None:
    has_mmlu = not df["mmlu"].isna().all()
    has_he = not df["humaneval"].isna().all() if "humaneval" in df else False

    if not has_mmlu:
        return

    st.markdown("### 🎯 Quality Scores")
    top_models = df.dropna(subset=["mmlu"]).nlargest(8, "mmlu")

    fig = go.Figure()
    for _, row in top_models.iterrows():
        vals = [row["mmlu"] * 100]
        cats = ["MMLU"]
        if has_he and pd.notna(row.get("humaneval")):
            vals.append(row["humaneval"] * 100)
            cats.append("HumanEval")
        vals.append(vals[0])  # close polygon
        cats.append(cats[0])

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats,
            name=row["name"],
            fill="toself",
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[40, 100])),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame) -> None:
    st.markdown("### 📊 Full Benchmark Results")

    display_cols = {
        "name": "Model",
        "params": "Params",
        "quantization": "Quant",
        "tokens_per_sec": "Tok/s",
        "ttft_ms": "TTFT (ms)",
        "memory_gb": "VRAM (GB)",
        "mmlu": "MMLU",
        "context_k": "Context (K)",
    }

    show_df = df[list(display_cols.keys())].rename(columns=display_cols).copy()
    if "MMLU" in show_df.columns:
        show_df["MMLU"] = show_df["MMLU"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
        )
    show_df["Tok/s"] = show_df["Tok/s"].apply(lambda x: f"{x:.1f}")
    show_df["TTFT (ms)"] = show_df["TTFT (ms)"].apply(lambda x: f"{x:.0f} ms")
    show_df["VRAM (GB)"] = show_df["VRAM (GB)"].apply(lambda x: f"{x:.1f} GB")

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name="llm_bench_results.csv",
        mime="text/csv",
    )


def render_model_catalog() -> None:
    st.markdown("### 📚 Model Catalog")
    st.markdown(f"**{len(MODELS)} models** in the registry across {len(get_families())} families")

    cols = st.columns(3)
    for i, (mid, meta) in enumerate(MODELS.items()):
        with cols[i % 3]:
            tags_html = " ".join(f'<span class="tag-pill">{t}</span>' for t in meta.get("tags", []))
            st.markdown(f"""
            <div style="background:#1a1a2e;border:1px solid #0f3460;border-radius:8px;padding:1rem;margin:0.5rem 0">
              <strong style="color:#e2e8f0">{meta['name']}</strong><br>
              <span style="color:#8892b0;font-size:0.8rem">{meta['developer']} · {meta['params']} · Apache/MIT</span><br>
              <span style="color:#64748b;font-size:0.75rem">ctx {meta['context']//1000}K · VRAM 4-bit: {meta['min_vram_gb'].get('4bit','?')} GB</span><br>
              <div style="margin-top:0.4rem">{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)


def render_live_benchmark_panel() -> None:
    st.markdown("---")
    st.markdown("### ▶️ Run Benchmarks on Your Hardware")
    st.info(
        "Live benchmarking loads models onto your GPU. "
        "Make sure you have enough VRAM before proceeding."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_models = st.multiselect(
            "Models to benchmark",
            options=list(MODELS.keys()),
            default=["llama-3.1-8b", "qwen2.5-7b"],
            format_func=lambda x: MODELS[x]["name"],
        )

    with col2:
        quantization = st.radio("Quantization", ["4bit", "8bit", "fp16"], index=0)

    with col3:
        benchmarks = st.multiselect(
            "Benchmarks",
            ["speed", "mmlu", "humaneval", "truthfulqa"],
            default=["speed"],
        )

    if st.button("🚀 Run Benchmarks", type="primary"):
        if not selected_models:
            st.warning("Please select at least one model.")
            return

        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, model_id in enumerate(selected_models):
            status.info(f"Loading {MODELS[model_id]['name']} ({quantization})…")
            try:
                from llm_bench.models.loader import ModelLoader, Quantization
                from llm_bench.benchmarks.speed import benchmark_speed

                quant_map = {"4bit": Quantization.INT4, "8bit": Quantization.INT8, "fp16": Quantization.FP16}
                loader = ModelLoader()
                result = loader.load(model_id, quantization=quant_map[quantization])

                if not result.success:
                    st.error(f"Failed to load {model_id}: {result.error}")
                    continue

                row: dict = {"model": MODELS[model_id]["name"], "quantization": quantization}

                if "speed" in benchmarks:
                    status.info(f"Speed benchmark: {MODELS[model_id]['name']}…")
                    sr = benchmark_speed(
                        result.model, result.tokenizer,
                        model_id=model_id, quantization=quantization,
                    )
                    row["tokens_per_sec"] = sr.tokens_per_second
                    row["ttft_ms"] = sr.time_to_first_token_ms
                    row["memory_gb"] = sr.memory_delta_gb

                if "mmlu" in benchmarks:
                    status.info(f"MMLU benchmark: {MODELS[model_id]['name']}…")
                    from llm_bench.benchmarks.quality import evaluate_mmlu
                    qr = evaluate_mmlu(result.model, result.tokenizer,
                                       model_id=model_id, quantization=quantization, num_samples=50)
                    row["mmlu"] = qr.score

                results.append(row)
                loader.unload(result)

            except Exception as exc:
                st.error(f"Error benchmarking {model_id}: {exc}")

            progress.progress((i + 1) / len(selected_models))

        status.success("✅ Benchmarks complete!")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    filters = render_sidebar()
    precomputed = load_precomputed()

    hw_tag = filters["hw_tag"]
    if hw_tag == "live":
        hw_tag = "rtx3090"  # fallback for display
        st.warning("Live hardware detection requires GPU + PyTorch. Showing RTX 3090 results as fallback.")

    # ── Hero header ────────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="font-family:JetBrains Mono,monospace;font-size:2.4rem;color:#e94560">🦙 llm-bench</h1>'
        '<p style="color:#8892b0;font-size:1.1rem;margin-top:-1rem">Compare 20+ local LLMs — speed, quality, and memory before you download</p>',
        unsafe_allow_html=True,
    )

    render_hardware_banner(hw_tag, precomputed)

    # ── Build / filter DataFrame ───────────────────────────────────────────
    df = build_results_df(hw_tag, precomputed)

    if filters["family_filter"] != "All":
        df = df[df["family"] == filters["family_filter"]]
    if filters["quant_filter"]:
        df = df[df["quantization"].isin(filters["quant_filter"])]
    df = df[df["tokens_per_sec"] >= filters["min_speed"]]
    if filters["min_mmlu"] > 0:
        df = df[df["mmlu"].fillna(0) >= filters["min_mmlu"]]

    # ── KPI row ────────────────────────────────────────────────────────────
    if not df.empty:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Models shown", len(df))
        kpi2.metric("Fastest (tok/s)", f"{df['tokens_per_sec'].max():.0f}")
        kpi3.metric("Best MMLU", f"{df['mmlu'].max()*100:.1f}%" if df["mmlu"].notna().any() else "—")
        kpi4.metric("Smallest VRAM", f"{df['memory_gb'].min():.1f} GB")

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_overview, tab_speed, tab_quality, tab_memory, tab_catalog, tab_live = st.tabs([
        "📊 Overview", "⚡ Speed", "🎯 Quality", "💾 Memory", "📚 Catalog", "▶️ Live Benchmark"
    ])

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            render_scatter(df)
        with c2:
            render_quality_radar(df)
        render_data_table(df)

    with tab_speed:
        render_speed_bars(df)
        if not df.empty:
            st.markdown("#### Time to First Token (TTFT)")
            ttft_df = df.sort_values("ttft_ms").head(15)
            fig2 = px.bar(ttft_df, x="name", y="ttft_ms", color="family",
                          template="plotly_dark", labels={"ttft_ms": "TTFT (ms)", "name": ""},
                          color_discrete_sequence=px.colors.qualitative.Bold)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig2, use_container_width=True)

    with tab_quality:
        render_quality_radar(df)
        if not df.empty and df["mmlu"].notna().any():
            st.markdown("#### MMLU Accuracy by Model")
            qdf = df.dropna(subset=["mmlu"]).sort_values("mmlu", ascending=False)
            fig3 = px.bar(qdf, x="name", y="mmlu", color="family",
                          template="plotly_dark", labels={"mmlu": "MMLU Accuracy", "name": ""},
                          color_discrete_sequence=px.colors.qualitative.Bold)
            fig3.update_layout(yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig3, use_container_width=True)

    with tab_memory:
        render_memory_chart(df)

    with tab_catalog:
        render_model_catalog()

    with tab_live:
        render_live_benchmark_panel()


if __name__ == "__main__":
    main()
