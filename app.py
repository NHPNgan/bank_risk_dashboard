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

DIAGNOSTIC_TIPS = {
    "E": [
        "Xu hướng biên lãi ròng (NIM) và cơ cấu thu nhập lãi vs phi lãi gần đây",
        "Chất lượng danh mục cho vay có gây áp lực lên lợi nhuận không",
        "So sánh ROE/ROA với trung bình ngành trong cùng giai đoạn",
    ],
    "M": [
        "Cơ cấu chi phí hoạt động (nhân sự, mạng lưới chi nhánh) có đang phình to bất thường",
        "Tỷ lệ chi phí/thu nhập (CIR) so với kế hoạch ngân sách nội bộ",
        "Hiệu quả đầu tư công nghệ/số hóa để giảm chi phí vận hành",
    ],
    "A": [
        "Mức độ tập trung tín dụng theo ngành/khách hàng lớn",
        "Xu hướng nợ xấu mới phát sinh (NPL formation) và các khoản nợ tái cơ cấu",
        "Chính sách trích lập dự phòng có đủ thận trọng so với rủi ro danh mục",
    ],
    "L": [
        "Mức độ phụ thuộc vào nguồn vốn bán buôn/liên ngân hàng ngắn hạn",
        "Độ tập trung tiền gửi (khách hàng lớn chiếm bao nhiêu % tổng huy động)",
        "Xu hướng tỷ lệ LDR và khả năng đáp ứng rút tiền đột biến",
    ],
    "C": [
        "Kế hoạch tăng vốn hoặc giữ lại lợi nhuận trong 1-2 năm tới",
        "Tốc độ tăng tài sản có rủi ro (RWA) so với tốc độ tăng vốn chủ sở hữu",
        "Chính sách cổ tức có đang bào mòn bộ đệm vốn không",
    ],
}

st.set_page_config(page_title="Bank Risk Early-Warning Dashboard", layout="wide")


@st.cache_data
def load_default():
    return run_pipeline(DATA_PATH)


@st.cache_data
def load_from_bytes(file_bytes, sheet_name):
    return run_pipeline(BytesIO(file_bytes), sheet_name=sheet_name)


# ---------------------------------------------------------------------------
# Sidebar - optional data upload (dùng để nạp dữ liệu cập nhật làm "test set"
# demo trực tiếp trong buổi thuyết trình, không cần sửa code / deploy lại)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Dữ liệu nguồn")
uploaded_file = st.sidebar.file_uploader(
    "Nạp file dữ liệu cập nhật (.xlsx, cùng cấu trúc cột với bộ dữ liệu gốc)",
    type=["xlsx"],
)
sheet_name = st.sidebar.text_input("Tên sheet chứa dữ liệu", value="Data")

if uploaded_file is not None:
    try:
        result = load_from_bytes(uploaded_file.getvalue(), sheet_name)
        st.sidebar.success(f"Đang dùng dữ liệu tải lên: **{uploaded_file.name}**")
        if st.sidebar.button("Quay lại dữ liệu mặc định"):
            uploaded_file = None
            st.rerun()
    except Exception as e:
        st.sidebar.error(
            f"Không đọc được file này (kiểm tra lại tên sheet và cấu trúc cột). Chi tiết lỗi: {e}"
        )
        st.sidebar.warning("Đang dùng tạm dữ liệu mặc định bên dưới.")
        result = load_default()
else:
    result = load_default()
    st.sidebar.caption("Đang dùng dữ liệu mặc định: Le et al. (2022), 2002-2021, 44 ngân hàng.")

labeled = result["labeled"]
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
# Header
# ---------------------------------------------------------------------------
st.title("Bank Risk Early-Warning Supervision Dashboard")
_n_banks = labeled["Bank Code"].nunique()
_yr_min, _yr_max = int(labeled["Year"].min()), int(labeled["Year"].max())
_data_desc = (
    f"file tải lên ({uploaded_file.name})" if uploaded_file is not None
    else "Le et al. (2022) Vietnamese banking dataset"
)
st.caption(
    f"Demo prototype - CAMEL block-wise PCA + K-means clustering. "
    f"Data: {_data_desc}, {_n_banks} ngân hàng, {_yr_min}-{_yr_max}. "
    f"Method reimplemented from Chung & Hung (2026), *Int. Review of Economics and Finance*, 110, 105571."
)

tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Bank Detail", "System Transitions"])

# ---------------------------------------------------------------------------
# TAB 1 - Portfolio Overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("System snapshot (latest year on record per bank)")

    counts = latest["RiskState"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    cols = st.columns(5)
    for col, state in zip(cols, RISK_ORDER):
        with col:
            st.markdown(
                f"""<div style="background:{STATUS_COLOR[state]}1a;border:1px solid {STATUS_COLOR[state]};
                border-radius:10px;padding:14px;text-align:center;">
                <div style="font-size:26px;font-weight:700;color:{STATUS_COLOR[state]}">{counts[state]}</div>
                <div style="font-size:13px;color:#52514e;">{state}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("#### Recent alerts - banks whose latest year-over-year move was a worsening transition")
    if alerts.empty:
        st.info(
            "Không có ngân hàng nào chuyển biến xấu ở cặp năm liên tiếp gần nhất trong dữ liệu lịch sử "
            "hiện có. Tính năng này sẽ có ý nghĩa rõ hơn khi nạp dữ liệu cập nhật làm test set."
        )
    else:
        for _, row in alerts.iterrows():
            st.markdown(
                f"""<div style="border-left:4px solid {STATUS_COLOR[row['To state']]};padding:8px 12px;
                margin-bottom:6px;background:#fcfcfb;border-radius:4px;">
                <b>{row['Bank Code']}</b> &nbsp; {row['From year']} → {row['To year']}: &nbsp;
                <span style="color:{STATUS_COLOR[row['From state']]}">{row['From state']}</span>
                &nbsp;→&nbsp;
                <span style="color:{STATUS_COLOR[row['To state']]};font-weight:700">{row['To state']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("#### Portfolio table")
    filter_states = st.multiselect("Lọc theo nhóm rủi ro", RISK_ORDER, default=RISK_ORDER)
    view = latest[latest["RiskState"].isin(filter_states)].copy()
    view["Primary driver"] = view["DominantFactor"].map(FACTOR_NAME)
    display_cols = ["Bank Code", "Year", "RiskState", "Primary driver"] + FACTOR_COLS

    def style_state(s):
        return [f"background-color:{STATUS_COLOR[v]}22;color:{STATUS_COLOR[v]};font-weight:600" for v in s]

    st.dataframe(
        view[display_cols].style.apply(style_state, subset=["RiskState"]).format(
            {c: "{:.2f}" for c in FACTOR_COLS}
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# TAB 2 - Bank Detail
# ---------------------------------------------------------------------------
with tab2:
    bank_list = sorted(labeled["Bank Code"].unique())
    default_idx = bank_list.index("ABB") if "ABB" in bank_list else 0
    bank = st.selectbox("Chọn ngân hàng", bank_list, index=default_idx)

    bh = labeled[labeled["Bank Code"] == bank].sort_values("Year")
    cur = bh.iloc[-1]

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(
            f"""<div style="background:{STATUS_COLOR[cur['RiskState']]}1a;border:1px solid {STATUS_COLOR[cur['RiskState']]};
            border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:13px;color:#52514e;">{bank} - {int(cur['Year'])}</div>
            <div style="font-size:20px;font-weight:700;color:{STATUS_COLOR[cur['RiskState']]}">{cur['RiskState']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        n_years = len(bh)
        first_state, last_state = bh.iloc[0]["RiskState"], bh.iloc[-1]["RiskState"]
        st.caption(
            f"Dữ liệu có {n_years} năm ({int(bh['Year'].min())}-{int(bh['Year'].max())}). "
            f"Trạng thái đầu chuỗi: **{first_state}** -> trạng thái gần nhất: **{last_state}**."
        )

    st.markdown("##### Xu hướng rủi ro theo thời gian")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=bh["Year"], y=bh["AggRisk"], mode="lines+markers",
        line=dict(color="#898781", width=2),
        marker=dict(size=11, color=[STATUS_COLOR[s] for s in bh["RiskState"]], line=dict(width=1, color="#fff")),
        text=bh["RiskState"], hovertemplate="Năm %{x}<br>Risk index: %{y:.2f}<br>%{text}<extra></extra>",
        showlegend=False,
    ))
    fig_trend.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Năm", yaxis_title="Aggregate risk index (risk-oriented)",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
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
        ))
        fig_bar.add_vline(x=0, line_color="#c3c2b7")
        fig_bar.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Deviation (risk-oriented; + = higher risk)",
            plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        same_year = labeled[labeled["Year"] == cur["Year"]]
        pct = (same_year["AggRisk"] < cur["AggRisk"]).mean() * 100
        st.caption(
            f"So với {len(same_year)} ngân hàng khác cùng năm {int(cur['Year'])}: "
            f"chỉ số rủi ro tổng hợp của {bank} cao hơn **{pct:.0f}%** ngân hàng trong mẫu "
            f"(percentile {pct:.0f} - 100 = rủi ro cao nhất hệ thống)."
        )

    with col_diag:
        st.markdown("##### Gợi ý hướng đào sâu (diagnostic)")
        dom = cur["DominantFactor"]
        st.markdown(f"Yếu tố nổi bật nhất: **{FACTOR_NAME[dom]}**")
        for tip in DIAGNOSTIC_TIPS[dom]:
            st.markdown(f"- {tip}")
        st.caption(
            "Đây là gợi ý chẩn đoán (nên xem thêm dữ liệu gì), không phải khuyến nghị hành động cụ thể - "
            "quyết định giám sát/đầu tư vẫn thuộc thẩm quyền chuyên môn của người dùng."
        )

# ---------------------------------------------------------------------------
# TAB 3 - System Transitions
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Ma trận chuyển trạng thái 1 năm (toàn hệ thống)")
    z = transition_probs.values
    fig_heat = go.Figure(go.Heatmap(
        z=z, x=RISK_ORDER, y=RISK_ORDER,
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        text=[[f"{v:.0%}" for v in row] for row in z],
        texttemplate="%{text}", hovertemplate="%{y} -> %{x}: %{z:.1%}<extra></extra>",
        colorbar=dict(title="P", tickformat=".0%"),
    ))
    fig_heat.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Trạng thái năm t+1", yaxis_title="Trạng thái năm t",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Độ 'dính' của từng trạng thái (persistence)")
    persistence = [transition_probs.loc[s, s] for s in RISK_ORDER]
    fig_pers = go.Figure(go.Bar(
        x=RISK_ORDER, y=persistence,
        marker_color=[STATUS_COLOR[s] for s in RISK_ORDER],
        text=[f"{p:.0%}" for p in persistence], textposition="outside",
    ))
    fig_pers.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Xác suất ở lại cùng trạng thái năm sau", yaxis_tickformat=".0%",
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    st.plotly_chart(fig_pers, use_container_width=True)
    st.caption(
        "Trạng thái có persistence cao (vd. Stressed) nghĩa là ngân hàng thường ở lại đó nhiều năm liền - "
        "phù hợp với phát hiện của bài báo gốc rằng rủi ro ngân hàng thay đổi từ từ, không đột ngột."
    )
