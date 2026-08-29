"""
Bank Risk Early-Warning Supervision Dashboard
Demo web app for a group course project.

Methodology: CAMEL block-wise PCA + K-means clustering, reimplemented from
Chung, N.H. & Hung, V.T. (2026), "Bank risk clustering and early warning
supervision," International Review of Economics and Finance, 110, 105571.

Run with: streamlit run app.py
"""
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pipeline import run_pipeline, RISK_ORDER, BLOCKS

# CAMEL indicators grouped by block (E, M, A, L, C) in display order, for the raw-ratio table
CAMEL_COLS_ORDER = [c for cols in BLOCKS.values() for c in cols]

# ---------------------------------------------------------------------------
# Config & palette (validated dataviz-skill palette; status colors for risk
# states, categorical colors for CAMEL factors, diverging blue<->red for
# deviation bars, sequential blue for the transition heatmap)
# ---------------------------------------------------------------------------
DATA_PATH = "data/VN_banks_dataset.xlsx"

STATUS_COLOR = {
    "Low risk": "#0ca30c",
    "Moderate": "#c9a227",
    "Watchlist": "#fab219",
    "Stressed": "#ec835a",
    "Distress": "#d03b3b",
}
# Redundant, non-color encoding for the 5 risk states (accessibility: don't rely on hue alone,
# e.g. for red-green color-blind viewers). Distinct shapes rather than a color-only legend.
RISK_ICON = {
    "Low risk": "●",   # ●
    "Moderate": "◆",   # ◆
    "Watchlist": "▲",  # ▲
    "Stressed": "■",   # ■
    "Distress": "✖",   # ✖
}
FACTOR_COLOR = {"E": "#2a78d6", "M": "#eb6834", "A": "#1baf7a", "L": "#eda100", "C": "#e87ba4"}
FACTOR_NAME = {
    "E": "Earnings (ROE, ROA, NIM)",
    "M": "Management (CIR, NIE)",
    "A": "Asset quality (NPLR, PCR)",
    "L": "Liquidity (LTD, LTA)",
    "C": "Capital (ETA, ETD)",
}
POS_COLOR, NEG_COLOR = "#e34948", "#2a78d6"   # diverging: positive=risk(red), negative=strength(blue)
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
BENCH_COLOR = "#52514e"

# Regulatory reference used purely as a visual benchmark line (not a model input).
# 3% is the NPL ceiling widely referenced in Vietnamese banking regulation as a
# condition of "safe" operating status (e.g. cited as the threshold credit
# institutions must stay under to be allowed to contribute capital / buy shares).
NPL_BENCHMARK = 3.0

RAW_LABEL = {
    "ROE": "ROE - Return on Equity", "ROA": "ROA - Return on Assets",
    "NIM": "NIM - Net Interest Margin", "CIR": "CIR - Cost-to-Income Ratio",
    "NIE": "NIE - Non-Interest Expense / Total Income", "NPLR": "NPLR - Non-Performing Loan Ratio",
    "PCR": "PCR - Provision Coverage Ratio", "LTD": "LTD - Liquid Assets / Total Deposits",
    "LTA": "LTA - Liquid Assets / Total Assets", "ETA": "ETA - Equity / Total Assets",
    "ETD": "ETD - Equity / Total Deposits",
}
RAW_BLOCK_OF = {c: b for b, cols in BLOCKS.items() for c in cols}

DIAGNOSTIC_TIPS = {
    "E": [
        "Recent NIM trend and the mix of interest vs. non-interest income",
        "Whether loan portfolio quality is putting pressure on profitability",
        "Compare ROE/ROA against the industry average for the same period",
    ],
    "M": [
        "Whether the cost structure (staff, branch network) is expanding unusually fast",
        "Cost-to-income ratio (CIR) against internal budget plans",
        "Effectiveness of technology/digitalization investment in reducing operating costs",
    ],
    "A": [
        "Credit concentration by sector or large borrowers",
        "Trend in new NPL formation and restructured/rescheduled loans",
        "Whether provisioning policy is conservative enough relative to portfolio risk",
    ],
    "L": [
        "Reliance on short-term wholesale/interbank funding",
        "Deposit concentration (share of total funding from large depositors)",
        "LDR trend and capacity to meet a sudden spike in withdrawals",
    ],
    "C": [
        "Capital-raising plans or profit retention over the next 1-2 years",
        "Growth rate of risk-weighted assets (RWA) relative to equity growth",
        "Whether dividend policy is eroding the capital buffer",
    ],
}

st.set_page_config(page_title="Bank Risk Early-Warning Dashboard", layout="wide")


@st.cache_data
def load_default():
    return run_pipeline(DATA_PATH)


@st.cache_data
def load_from_bytes(file_bytes, sheet_name):
    return run_pipeline(BytesIO(file_bytes), sheet_name=sheet_name)


def _friendly_upload_error(e: Exception, sheet: str) -> str:
    """Translate a raw pandas/pipeline exception into plain-English guidance."""
    msg = str(e)
    if isinstance(e, KeyError):
        return (
            f"This file is missing an expected column ({msg}). Make sure the column "
            f"headers exactly match the original dataset template."
        )
    if "Worksheet" in msg or "sheet" in msg.lower():
        return (
            f'Could not find a sheet named "{sheet}" in this file. Check the sheet '
            f"name field on the left and try again."
        )
    return (
        "Could not read this file. Please check that it uses the same sheet name and "
        "column headers as the original dataset template."
    )


def _goto_bank(bank_code):
    """Callback used by 'View' buttons to jump straight to Bank Detail for one bank."""
    st.session_state.page = "Bank Detail"
    st.session_state.bank_select = bank_code


# ---------------------------------------------------------------------------
# Sidebar - optional data upload (lets the user load updated data as a live
# "test set" during the presentation, with no code edit or redeploy needed)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Data source")
uploaded_file = st.sidebar.file_uploader(
    "Upload updated data (.xlsx, same column structure as the original dataset)",
    type=["xlsx"],
)
sheet_name = st.sidebar.text_input("Sheet name containing the data", value="Data")

if uploaded_file is not None:
    try:
        result = load_from_bytes(uploaded_file.getvalue(), sheet_name)
        st.sidebar.success(f"Using uploaded data: **{uploaded_file.name}**")
        if st.sidebar.button("Reset to default data"):
            uploaded_file = None
            st.rerun()
    except Exception as e:
        st.sidebar.error(_friendly_upload_error(e, sheet_name))
        with st.sidebar.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}")
        st.sidebar.warning("Falling back to the default dataset below.")
        result = load_default()
else:
    result = load_default()
    st.sidebar.caption("Using default dataset: Le et al. (2022), 2002-2021, 44 banks.")

labeled = result["labeled"]
cleaned = result["cleaned"]  # raw CAMEL ratios (%), before risk-orientation/standardization
centroids = result["centroids"]
transition_probs = result["transition_probs"]
timelines = result["timelines"]
alerts = result["alerts"]
block_pca_info = result["block_pca_info"]

FACTOR_COLS = ["E", "M", "A", "L", "C"]
labeled["AggRisk"] = labeled[FACTOR_COLS].mean(axis=1)
labeled["DominantFactor"] = labeled[FACTOR_COLS].idxmax(axis=1)

# latest snapshot per bank (for the portfolio table)
latest = labeled.sort_values("Year").groupby("Bank Code").tail(1).reset_index(drop=True)
latest = latest.sort_values("AggRisk", ascending=False)

# ---------------------------------------------------------------------------
# Header + navigation
# (a horizontal radio, not st.tabs, so a "View" button elsewhere in the app
# can jump the user straight to Bank Detail with a bank pre-selected)
# ---------------------------------------------------------------------------
st.title("Bank Risk Early-Warning Supervision Dashboard")

if "page" not in st.session_state:
    st.session_state.page = "Portfolio Overview"

page = st.radio(
    "Navigate", ["Portfolio Overview", "Bank Detail", "System Transitions"],
    horizontal=True, key="page", label_visibility="collapsed",
)
st.markdown("---")

# ---------------------------------------------------------------------------
# PAGE - Portfolio Overview
# ---------------------------------------------------------------------------
if page == "Portfolio Overview":
    st.subheader("System snapshot (latest year on record per bank)")

    counts = latest["RiskState"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    cols = st.columns(5)
    for col, state in zip(cols, RISK_ORDER):
        with col:
            st.markdown(
                f"""<div style="background:{STATUS_COLOR[state]}1a;border:1px solid {STATUS_COLOR[state]};
                border-radius:10px;padding:14px;text-align:center;">
                <div style="font-size:26px;font-weight:700;color:{STATUS_COLOR[state]}">{counts[state]}</div>
                <div style="font-size:13px;color:#52514e;">{RISK_ICON[state]} {state}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("#### System-wide risk trend over time")
    yearly = (
        labeled.groupby(["Year", "RiskState"]).size().unstack(fill_value=0)
        .reindex(columns=RISK_ORDER, fill_value=0).sort_index()
    )
    fig_sys = go.Figure()
    for state in RISK_ORDER:
        fig_sys.add_trace(go.Bar(
            x=yearly.index, y=yearly[state], name=f"{RISK_ICON[state]} {state}",
            marker_color=STATUS_COLOR[state],
            hovertemplate=f"%{{x}}<br>{state}: %{{y}} banks<extra></extra>",
        ))
    fig_sys.update_layout(
        barmode="stack", height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Year", yaxis_title="Number of banks", legend_title="Risk group",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    st.plotly_chart(fig_sys, use_container_width=True)
    st.caption(
        "Number of banks in each risk group by year - useful for spotting periods where the "
        "whole system worsened (red/orange area expands) versus just a few individual banks."
    )

    st.markdown("#### Recent alerts - banks whose latest year-over-year move was a worsening transition")
    if alerts.empty:
        st.info(
            "No bank had a worsening transition between the latest consecutive years in the "
            "current historical data. This feature becomes more meaningful once updated data "
            "is loaded as a live test set."
        )
    else:
        for _, row in alerts.iterrows():
            col_alert, col_btn = st.columns([6, 1])
            with col_alert:
                st.markdown(
                    f"""<div style="border-left:4px solid {STATUS_COLOR[row['To state']]};padding:8px 12px;
                    margin-bottom:6px;background:#fcfcfb;border-radius:4px;">
                    <b>{row['Bank Code']}</b> &nbsp; {row['From year']} → {row['To year']}: &nbsp;
                    <span style="color:{STATUS_COLOR[row['From state']]}">{RISK_ICON[row['From state']]} {row['From state']}</span>
                    &nbsp;→&nbsp;
                    <span style="color:{STATUS_COLOR[row['To state']]};font-weight:700">{RISK_ICON[row['To state']]} {row['To state']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col_btn:
                st.button(
                    "View →", key=f"goto_{row['Bank Code']}_{row['To year']}",
                    on_click=_goto_bank, args=(row["Bank Code"],),
                )

    st.markdown("#### Portfolio table")
    filter_states = st.multiselect("Filter by risk group", RISK_ORDER, default=RISK_ORDER)
    view = latest[latest["RiskState"].isin(filter_states)].copy()
    view["Primary driver"] = view["DominantFactor"].map(FACTOR_NAME)
    view["Risk state"] = view["RiskState"].map(lambda s: f"{RISK_ICON[s]} {s}")
    display_cols = ["Bank Code", "Year", "Risk state", "Primary driver"] + FACTOR_COLS

    def style_state(s):
        return [
            f"background-color:{STATUS_COLOR[v.split(' ', 1)[1]]}22;"
            f"color:{STATUS_COLOR[v.split(' ', 1)[1]]};font-weight:600"
            for v in s
        ]

    st.dataframe(
        view[display_cols].style.apply(style_state, subset=["Risk state"]).format(
            {c: "{:.2f}" for c in FACTOR_COLS}
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# PAGE - Bank Detail
# ---------------------------------------------------------------------------
elif page == "Bank Detail":
    bank_list = sorted(labeled["Bank Code"].unique())
    if "bank_select" not in st.session_state or st.session_state.bank_select not in bank_list:
        st.session_state.bank_select = "ABB" if "ABB" in bank_list else bank_list[0]
    bank = st.selectbox("Select bank", bank_list, key="bank_select")

    bh = labeled[labeled["Bank Code"] == bank].sort_values("Year")
    cur = bh.iloc[-1]

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(
            f"""<div style="background:{STATUS_COLOR[cur['RiskState']]}1a;border:1px solid {STATUS_COLOR[cur['RiskState']]};
            border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:13px;color:#52514e;">{bank} - {int(cur['Year'])}</div>
            <div style="font-size:20px;font-weight:700;color:{STATUS_COLOR[cur['RiskState']]}">{RISK_ICON[cur['RiskState']]} {cur['RiskState']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        n_years = len(bh)
        first_state, last_state = bh.iloc[0]["RiskState"], bh.iloc[-1]["RiskState"]
        st.caption(
            f"Data spans {n_years} years ({int(bh['Year'].min())}-{int(bh['Year'].max())}). "
            f"Starting state: **{first_state}** → latest state: **{last_state}**."
        )

    st.markdown("##### Risk trend over time")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=bh["Year"], y=bh["AggRisk"], mode="lines+markers",
        line=dict(color="#898781", width=2),
        marker=dict(size=11, color=[STATUS_COLOR[s] for s in bh["RiskState"]], line=dict(width=1, color="#fff")),
        text=bh["RiskState"], hovertemplate="Year %{x}<br>Risk index: %{y:.2f}<br>%{text}<extra></extra>",
        showlegend=False,
    ))
    fig_trend.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Year", yaxis_title="Aggregate risk index (risk-oriented)",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    fig_trend.update_xaxes(dtick=1, tickformat="d")  # whole years only, never fractional ticks
    st.plotly_chart(fig_trend, use_container_width=True)

    col_bar, col_diag = st.columns([1.3, 1])
    with col_bar:
        st.markdown(f"##### CAMEL factor breakdown - {int(cur['Year'])}")
        vals = [cur[c] for c in FACTOR_COLS]
        colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]
        fig_bar = go.Figure(go.Bar(
            x=vals, y=[FACTOR_NAME[c] for c in FACTOR_COLS], orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in vals], textposition="outside",
            cliponaxis=False,  # never clip the value label even if it overhangs the plot area
        ))
        fig_bar.add_vline(x=0, line_color="#c3c2b7")
        # Pad the x-axis range so an "outside" text label always has room, instead of
        # being cut off at the edge of the plot for a large positive/negative bar.
        _vmin, _vmax = min(vals + [0]), max(vals + [0])
        _pad = max(abs(_vmin), abs(_vmax)) * 0.35 + 0.05
        fig_bar.update_xaxes(range=[_vmin - _pad, _vmax + _pad])
        fig_bar.update_yaxes(automargin=True)
        fig_bar.update_layout(
            height=260, margin=dict(l=10, r=20, t=10, b=10),
            xaxis_title="Deviation (risk-oriented; + = higher risk)",
            plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        same_year = labeled[labeled["Year"] == cur["Year"]]
        pct = (same_year["AggRisk"] < cur["AggRisk"]).mean() * 100
        st.caption(
            f"Compared to {len(same_year)} other banks in {int(cur['Year'])}: {bank}'s aggregate "
            f"risk index is higher than **{pct:.0f}%** of banks in the sample "
            f"(percentile {pct:.0f} - 100 = highest risk in the system)."
        )

    with col_diag:
        st.markdown("##### Suggested areas to investigate (diagnostic)")
        dom = cur["DominantFactor"]
        st.markdown(f"Dominant factor: **{FACTOR_NAME[dom]}**")
        for tip in DIAGNOSTIC_TIPS[dom]:
            st.markdown(f"- {tip}")
        st.caption(
            "These are diagnostic suggestions (what to look into further), not specific "
            "supervisory or investment recommendations - that decision remains within the "
            "user's own professional judgment."
        )

    st.markdown("---")
    raw_bh = cleaned[cleaned["Bank Code"] == bank].sort_values("Year")
    raw_cur = raw_bh[raw_bh["Year"] == cur["Year"]].iloc[0]

    col_raw, col_npl = st.columns([1, 1.3])
    with col_raw:
        st.markdown(f"##### Actual CAMEL ratios (%) - {int(cur['Year'])}")
        raw_table = pd.DataFrame({
            "Indicator": [RAW_LABEL[c] for c in CAMEL_COLS_ORDER],
            "Block": [RAW_BLOCK_OF[c] for c in CAMEL_COLS_ORDER],
            "Value (%)": [raw_cur[c] for c in CAMEL_COLS_ORDER],
        })

        def style_block(s):
            return [f"background-color:{FACTOR_COLOR[v]}22;color:{FACTOR_COLOR[v]};font-weight:600" for v in s]

        st.dataframe(
            raw_table.style.apply(style_block, subset=["Block"]).format({"Value (%)": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Original financial ratios (before risk-orientation / standardization) - for "
            "comparing directly against regulatory thresholds or reported figures, rather "
            "than only the standardized deviation shown above."
        )

    with col_npl:
        st.markdown("##### NPL ratio over time - vs. reference benchmark")
        fig_npl = go.Figure()
        fig_npl.add_trace(go.Scatter(
            x=raw_bh["Year"], y=raw_bh["NPLR"], mode="lines+markers", name=bank,
            line=dict(color=STATUS_COLOR["Stressed"], width=2), marker=dict(size=9),
            hovertemplate="Year %{x}<br>NPL: %{y:.2f}%<extra></extra>",
        ))
        fig_npl.add_hline(
            y=NPL_BENCHMARK, line_dash="dash", line_color=BENCH_COLOR,
            annotation_text=f"Reference benchmark {NPL_BENCHMARK:.0f}%", annotation_position="top left",
        )
        fig_npl.update_layout(
            height=290, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Year", yaxis_title="NPL (%)",
            plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        )
        fig_npl.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig_npl, use_container_width=True)
        st.caption(
            f"Dashed line: {NPL_BENCHMARK:.0f}% - the on-balance-sheet NPL ratio commonly used "
            "as a safe-operation threshold in Vietnamese banking regulation (a visual reference "
            "only, not an input to the clustering model)."
        )

# ---------------------------------------------------------------------------
# PAGE - System Transitions
# ---------------------------------------------------------------------------
elif page == "System Transitions":
    st.subheader("One-year transition matrix (system-wide)")
    axis_labels = [f"{RISK_ICON[s]} {s}" for s in RISK_ORDER]
    z = transition_probs.values
    fig_heat = go.Figure(go.Heatmap(
        z=z, x=axis_labels, y=axis_labels,
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        text=[[f"{v:.0%}" for v in row] for row in z],
        texttemplate="%{text}", hovertemplate="%{y} -> %{x}: %{z:.1%}<extra></extra>",
        colorbar=dict(title="P", tickformat=".0%"),
    ))
    fig_heat.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="State at year t+1", yaxis_title="State at year t",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader('State persistence ("stickiness")')
    persistence = [transition_probs.loc[s, s] for s in RISK_ORDER]
    fig_pers = go.Figure(go.Bar(
        x=axis_labels, y=persistence,
        marker_color=[STATUS_COLOR[s] for s in RISK_ORDER],
        text=[f"{p:.0%}" for p in persistence], textposition="outside",
        cliponaxis=False,
    ))
    fig_pers.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Probability of remaining in the same state next year", yaxis_tickformat=".0%",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    st.plotly_chart(fig_pers, use_container_width=True)
    st.caption(
        "A high-persistence state (e.g. Stressed) means banks tend to stay there for several "
        "consecutive years - consistent with the original paper's finding that bank risk "
        "changes gradually rather than abruptly."
    )
