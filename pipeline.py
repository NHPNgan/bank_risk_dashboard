"""
Bank Risk Early-Warning pipeline
Reimplements the CAMEL block-wise PCA + K-means clustering methodology from
Chung, N.H. & Hung, V.T. (2026), "Bank risk clustering and early warning
supervision," International Review of Economics and Finance, 110, 105571.

Usage:
    from pipeline import run_pipeline
    result = run_pipeline("path/to/raw_dataset.xlsx")
    result["labeled"]        -> DataFrame: Bank Code, Year, E, M, A, L, C, RiskState
    result["centroids"]      -> DataFrame: cluster centroid profile
    result["transition_probs"] -> DataFrame: one-year transition matrix
    result["timelines"]      -> dict: Bank Code -> [(Year, RiskState), ...]
    result["alerts"]         -> DataFrame: banks whose latest move was a worsening transition
    result["block_pca_info"] -> dict: per-block loadings + explained variance
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

CAMEL_COLS = ["ROE", "ROA", "NIM", "CIR", "NIE", "NPLR", "PCR", "LTD", "LTA", "ETA", "ETD"]
INVERT = ["ROE", "ROA", "NIM", "PCR", "ETA", "ETD", "LTD", "LTA"]   # higher originally = safer
KEEP = ["CIR", "NIE", "NPLR"]                                       # higher originally = riskier
BLOCKS = {
    "E": ["ROE", "ROA", "NIM"],
    "M": ["CIR", "NIE"],
    "A": ["NPLR", "PCR"],
    "L": ["LTD", "LTA"],
    "C": ["ETA", "ETD"],
}
RISK_ORDER = ["Low risk", "Moderate", "Watchlist", "Stressed", "Distress"]
RISK_RANK = {s: i for i, s in enumerate(RISK_ORDER)}

RENAME_MAP = {
    "Returns Over Equity": "ROE", "Returns Over Assets": "ROA",
    "Net Interest Margin": "NIM", "Cost-Income Ratios": "CIR",
    "Non-performing Loans Ratio": "NPLR",
    "Liquid Assets Over Total Deposits": "LTD",
    "Liquid Assets Over Total Assets": "LTA",
    "Equity Over Total Assets": "ETA", "Equity Over Total Deposits": "ETD",
}


def load_raw(path, sheet_name="Data"):
    """Load the raw Le et al. (2022)-style workbook, dropping the abbreviation row
    and coercing everything except Bank Code to numeric."""
    raw = pd.read_excel(path, sheet_name=sheet_name)
    raw = raw.iloc[1:].reset_index(drop=True)
    for c in raw.columns:
        if c != "Bank Code":
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw


def clean_camel(raw, exclude_banks=("VBSP",)):
    """Compute the 11 CAMEL indicators and drop rows with any missing value."""
    df = raw[~raw["Bank Code"].isin(exclude_banks)].reset_index(drop=True).copy()

    df["NIE"] = (df["Non-Interest Expenses"] / df["Total Income"]) * 100
    df["PCR"] = (df["Loan Loss Provisions  "] / df["Non-performing Loans  "]) * 100
    df["PCR"] = df["PCR"].replace([np.inf, -np.inf], np.nan)

    df = df.rename(columns=RENAME_MAP)
    df = df[["Bank Code", "Year"] + CAMEL_COLS].copy()
    df = df.dropna(subset=CAMEL_COLS).reset_index(drop=True)
    df["Year"] = df["Year"].astype(int)
    return df


def log_modulus(x):
    return np.sign(x) * np.log1p(np.abs(x))


def risk_orient_and_standardize(df):
    """Flip sign of 'higher=safer' indicators, log-modulus transform, then z-score."""
    out = df.copy()
    for c in INVERT:
        out[c] = -out[c]
    for c in CAMEL_COLS:
        out[c] = log_modulus(out[c])
    means, stds = out[CAMEL_COLS].mean(), out[CAMEL_COLS].std()
    out[CAMEL_COLS] = (out[CAMEL_COLS] - means) / stds
    return out


def compute_factor_scores(df_std):
    """Block-wise PCA -> PC1 per CAMEL block, sign-oriented so higher = more risk."""
    factor_scores = pd.DataFrame(index=df_std.index)
    block_info = {}
    for block, cols in BLOCKS.items():
        X = df_std[cols].values
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X).flatten()
        avg = X.mean(axis=1)
        if np.corrcoef(pc1, avg)[0, 1] < 0:
            pc1 = -pc1
            pca.components_ = -pca.components_
        factor_scores[block] = pc1
        block_info[block] = {
            "indicators": cols,
            "explained_var_pc1": float(pca.explained_variance_ratio_[0]),
            "loadings": dict(zip(cols, pca.components_[0].round(3))),
        }
    out = pd.concat([df_std[["Bank Code", "Year"]], factor_scores], axis=1)
    return out, block_info


def cluster_banks(factor_df, k=5, random_state=42):
    """K-means on the 5D factor space, clusters ordered/labeled by aggregate risk."""
    factor_cols = ["E", "M", "A", "L", "C"]
    X = factor_df[factor_cols].values
    km = KMeans(n_clusters=k, n_init=50, random_state=random_state)
    raw_labels = km.fit_predict(X)
    centroids = km.cluster_centers_
    agg_risk = centroids.mean(axis=1)
    order = np.argsort(agg_risk)
    label_map = {raw_id: RISK_ORDER[rank] for rank, raw_id in enumerate(order)}

    out = factor_df.copy()
    out["RiskState"] = [label_map[c] for c in raw_labels]

    centroid_table = pd.DataFrame(centroids, columns=factor_cols)
    centroid_table["AggRiskIndex"] = agg_risk
    centroid_table["n"] = pd.Series(raw_labels).value_counts().reindex(range(k)).values
    centroid_table["RiskState"] = [label_map[i] for i in range(k)]
    centroid_table = centroid_table.sort_values("AggRiskIndex").reset_index(drop=True)
    return out, centroid_table


def compute_transitions(labeled_df):
    """One-year transition matrix, per-bank timelines, and 'recent worsening' alerts."""
    df = labeled_df.sort_values(["Bank Code", "Year"]).reset_index(drop=True)

    pairs = []
    for bank, g in df.groupby("Bank Code"):
        g = g.sort_values("Year")
        years, states = g["Year"].values, g["RiskState"].values
        for i in range(len(g) - 1):
            if years[i + 1] == years[i] + 1:
                pairs.append((states[i], states[i + 1]))
    trans_counts = pd.crosstab(
        pd.Series([p[0] for p in pairs]), pd.Series([p[1] for p in pairs])
    ).reindex(index=RISK_ORDER, columns=RISK_ORDER, fill_value=0)
    trans_probs = trans_counts.div(trans_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).round(3)

    timelines = {
        bank: list(zip(g.sort_values("Year")["Year"].astype(int), g.sort_values("Year")["RiskState"]))
        for bank, g in df.groupby("Bank Code")
    }

    alerts = []
    for bank, g in df.groupby("Bank Code"):
        g = g.sort_values("Year")
        years, states = g["Year"].values, g["RiskState"].values
        if len(g) >= 2 and years[-1] == years[-2] + 1:
            prev_rank, curr_rank = RISK_RANK[states[-2]], RISK_RANK[states[-1]]
            if curr_rank > prev_rank:
                alerts.append({
                    "Bank Code": bank, "From year": int(years[-2]), "To year": int(years[-1]),
                    "From state": states[-2], "To state": states[-1],
                    "Steps worsened": curr_rank - prev_rank,
                })
    alerts_df = pd.DataFrame(alerts)
    if not alerts_df.empty:
        alerts_df = alerts_df.sort_values("Steps worsened", ascending=False).reset_index(drop=True)

    return trans_probs, timelines, alerts_df


def run_pipeline(path, sheet_name="Data", exclude_banks=("VBSP",), k=5):
    raw = load_raw(path, sheet_name=sheet_name)
    cleaned = clean_camel(raw, exclude_banks=exclude_banks)
    standardized = risk_orient_and_standardize(cleaned)
    factor_scores, block_pca_info = compute_factor_scores(standardized)
    labeled, centroids = cluster_banks(factor_scores, k=k)
    transition_probs, timelines, alerts = compute_transitions(labeled)
    return {
        "cleaned": cleaned,
        "labeled": labeled,
        "centroids": centroids,
        "transition_probs": transition_probs,
        "timelines": timelines,
        "alerts": alerts,
        "block_pca_info": block_pca_info,
    }


if __name__ == "__main__":
    SRC = "/root/.claude/uploads/0332ede3-33b6-5c67-b234-ca42629d2690/c346d710-VN_banks_dataset_updated_August_2022__data_only.xlsx"
    result = run_pipeline(SRC)
    print("Rows:", len(result["labeled"]), "| Banks:", result["labeled"]["Bank Code"].nunique())
    print("\nCentroids:\n", result["centroids"].round(3))
    print("\nRiskState counts:\n", result["labeled"]["RiskState"].value_counts().reindex(RISK_ORDER))
