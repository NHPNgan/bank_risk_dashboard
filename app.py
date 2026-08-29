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
        "Dig into the NIM trend and the interest vs. non-interest income mix to pinpoint where earnings pressure is coming from.",
        "Cross-check with the Asset quality block - deteriorating loan quality is a common hidden driver of profitability pressure.",
        "Benchmark ROE and ROA against peer banks for the same period to see if this is bank-specific or sector-wide.",
    ],
    "M": [
        "Review whether staff and branch-network costs are growing faster than revenue - flag for a cost-efficiency deep dive.",
        "Compare the Cost-to-Income Ratio (CIR) against the bank's internal budget plan to catch any overspend early.",
        "Assess whether technology/digitalization investment is actually converting into lower operating costs.",
    ],
    "A": [
        "Map credit concentration by sector and large borrowers to identify where a single default would do the most damage.",
        "Track new NPL formation and restructured/rescheduled loans period-over-period to catch deterioration early.",
        "Request a provisioning-adequacy review if coverage looks thin relative to current portfolio risk.",
    ],
    "L": [
        "Quantify reliance on short-term wholesale/interbank funding - flag if it's climbing quickly.",
        "Check deposit concentration among large depositors; a handful of withdrawals could strain funding fast.",
        "Stress-test the LDR against a sudden spike in withdrawals to confirm the buffer can absorb it.",
    ],
    "C": [
        "Request an update on capital-raising plans and profit-retention targets for the next 1-2 years.",
        "Compare RWA growth to equity growth - flag if risk-weighted assets are outpacing the capital base.",
        "Review whether the dividend policy is eroding the capital buffer faster than it can be rebuilt.",
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

    st.markdown("#### Portfolio table")
    year_options = sorted(labeled["Year"].unique(), reverse=True)
    bank_options = sorted(labeled["Bank Code"].unique())

    filt_state, filt_year, filt_bank = st.columns(3)
    with filt_state:
        filter_states = st.multiselect("Filter by risk group", RISK_ORDER, default=RISK_ORDER)
    with filt_year:
        filter_years = st.multiselect("Filter by year", year_options, default=[year_options[0]])
    with filt_bank:
        filter_banks = st.multiselect("Filter by bank", bank_options, default=bank_options)

    view = labeled[
        labeled["RiskState"].isin(filter_states)
        & labeled["Year"].isin(filter_years)
        & labeled["Bank Code"].isin(filter_banks)
    ].copy()
    view = view.sort_values(["Year", "AggRisk"], ascending=[False, False])
    view["Primary driver"] = view["DominantFactor"].map(FACTOR_NAME)
    view["Risk state"] = view["RiskState"].map(lambda s: f"{RISK_ICON[s]} {s}")
    display_cols = ["Bank Code", "Year", "Risk state", "Primary driver"] + FACTOR_COLS

    def style_state(s):
        return [
            f"background-color:{STATUS_COLOR[v.split(' ', 1)[1]]}22;"
            f"color:{STATUS_COLOR[v.split(' ', 1)[1]]};font-weight:600"
            for v in s
        ]

    if view.empty:
        st.info("No rows match the current filter selection - adjust the risk group, year, or bank filters above.")
    else:
        st.dataframe(
            view[display_cols].style.apply(style_state, subset=["Risk state"]).format(
                {c: "{:.2f}" for c in FACTOR_COLS}
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Defaults to the latest year on record for each bank. Select additional years above to "
        "pull in historical rows for the same banks."
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

    st.markdown("---")
    detail_years = sorted(bh["Year"].unique(), reverse=True)
    detail_year = st.selectbox(
        "Year for CAMEL factor breakdown & Actual CAMEL ratios below",
        detail_years, index=0, key="bank_detail_year",
    )
    detail_row = bh[bh["Year"] == detail_year].iloc[0]

    col_bar, col_diag = st.columns([1.3, 1])
    with col_bar:
        st.markdown(f"##### CAMEL factor breakdown - {int(detail_row['Year'])}")
        vals = [detail_row[c] for c in FACTOR_COLS]
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

        same_year = labeled[labeled["Year"] == detail_row["Year"]]
        pct = (same_year["AggRisk"] < detail_row["AggRisk"]).mean() * 100
        st.caption(
            f"Compared to {len(same_year)} other banks in {int(detail_row['Year'])}: {bank}'s aggregate "
            f"risk index is higher than **{pct:.0f}%** of banks in the sample "
            f"(percentile {pct:.0f} - 100 = highest risk in the system)."
        )

    with col_diag:
        st.markdown("##### Suggested areas to investigate (diagnostic)")
        dom = detail_row["DominantFactor"]
        st.markdown(
            f"**{bank}'s risk profile in {int(detail_row['Year'])} is being driven mainly by "
            f"{FACTOR_NAME[dom]}. Here's where to focus next:**"
        )
        for tip in DIAGNOSTIC_TIPS[dom]:
            st.markdown(f"- {tip}")
        st.caption(
            "These are starting points for further diligence, not supervisory or investment "
            "conclusions - the final call remains with the reviewing officer."
        )

    st.markdown("---")
    raw_bh = cleaned[cleaned["Bank Code"] == bank].sort_values("Year")
    raw_cur = raw_bh[raw_bh["Year"] == detail_row["Year"]].iloc[0]

    st.markdown(f"##### Actual CAMEL ratios (%) - {int(detail_row['Year'])}")
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
        height=(len(raw_table) + 1) * 36 + 3,
    )
    st.caption(
        "Original financial ratios (before risk-orientation / standardization) - for "
        "comparing directly against regulatory thresholds or reported figures, rather "
        "than only the standardized deviation shown above."
    )
    st.caption(
        "Note on reading this alongside the chart above: the CAMEL factor breakdown "
        "benchmarks each block against the full 2002-2021 panel average - the same basis "
        "the underlying risk model uses - not against this single year's raw figures. So a "
        "block that reads as a relative strength there (a negative, lower-risk deviation) "
        "can still look elevated or subdued in the plain percentages below, and vice versa. "
        "That is not a contradiction - it reflects two different reference points: long-run "
        "system history versus this one year's actual numbers."
    )

# ---------------------------------------------------------------------------
# PAGE - System Transitions
# ---------------------------------------------------------------------------
elif page == "System Transitions":
    sys_bank_options = ["All banks (system-wide)"] + sorted(labeled["Bank Code"].unique())
    sys_filter_bank = st.selectbox("Filter by bank", sys_bank_options, key="sys_trans_bank")

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

    if sys_filter_bank != "All banks (system-wide)":
        st.markdown("---")
        st.subheader(f"{sys_filter_bank}'s own year-by-year transitions")
        tl = timelines.get(sys_filter_bank, [])
        if len(tl) < 2:
            st.info(f"Not enough historical data to show year-over-year transitions for {sys_filter_bank}.")
        else:
            rows = []
            for (y0, s0), (y1, s1) in zip(tl[:-1], tl[1:]):
                if RISK_ORDER.index(s1) > RISK_ORDER.index(s0):
                    change = "Worsened"
                elif RISK_ORDER.index(s1) < RISK_ORDER.index(s0):
                    change = "Improved"
                else:
                    change = "No change"
                rows.append({
                    "From year": int(y0), "From state": f"{RISK_ICON[s0]} {s0}",
                    "To year": int(y1), "To state": f"{RISK_ICON[s1]} {s1}",
                    "Change": change,
                })
            tl_df = pd.DataFrame(rows)

            _change_color = {
                "Worsened": STATUS_COLOR["Distress"], "Improved": STATUS_COLOR["Low risk"], "No change": "#898781",
            }

            def style_change(s):
                return [f"color:{_change_color[v]};font-weight:600" for v in s]

            st.dataframe(
                tl_df.style.apply(style_change, subset=["Change"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                f"{sys_filter_bank}'s own sequence of year-over-year risk-state moves, drawn directly "
                "from its history - useful for checking whether this bank's pattern matches or diverges "
                "from the system-wide tendencies shown above."
            )
