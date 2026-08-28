# Bank Risk Early-Warning Supervision Dashboard (demo)

Reimplements the CAMEL block-wise PCA + K-means clustering methodology from
Chung, N.H. & Hung, V.T. (2026), "Bank risk clustering and early warning
supervision," *International Review of Economics and Finance*, 110, 105571.

## Files

- `pipeline.py` — data cleaning, CAMEL ratio construction, risk-orientation +
  log-modulus transform + standardization, block-wise PCA, K-means clustering,
  transition-matrix / timeline / alert computation. All reusable functions.
- `app.py` — Streamlit dashboard (3 tabs: Portfolio Overview, Bank Detail,
  System Transitions). Imports `pipeline.py`.
- `requirements.txt` — Python dependencies.
- `camel_clean.csv`, `camel_standardized.csv`, `camel_factor_scores.csv`,
  `camel_clustered.csv`, `transition_matrix.csv`, `bank_timelines.json`,
  `recent_worsening_alerts.csv` — intermediate outputs saved while building
  the pipeline step by step (kept for reference / debugging; `app.py` does
  not read these, it re-runs `pipeline.run_pipeline()` on the raw source file).

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## Using updated/current data

The app has a **file uploader in the sidebar** ("Nạp file dữ liệu cập nhật") —
no code edits or redeploy needed. Upload a new `.xlsx` file with the same
column layout as the Le et al. (2022) Vietnamese banking dataset (Bank Code,
Year, plus the raw balance-sheet/income-statement fields the pipeline maps to
the 11 CAMEL indicators), confirm the sheet name (default `Data`), and the
whole dashboard — clustering, transitions, alerts — re-fits on that file
live. Since PCA + K-means aren't a pretrained model but a fit-on-whatever-
data-you-give-it procedure, this "retrains" correctly every time; there is no
separate training step. Use the "Quay lại dữ liệu mặc định" button to switch
back to the original 2002-2021 dataset. This is the intended way to demo a
live "test set" during the presentation.

If a file fails to load (wrong sheet name, missing/renamed columns), the app
shows the error and falls back to the default dataset rather than crashing.

For a permanent change to the default dataset instead, replace
`data/VN_banks_dataset.xlsx` in the repo (same file name) or edit `DATA_PATH`
at the top of `app.py`.

**Important:** since PCA + K-means refit fresh on whatever rows are given,
uploading only a few recent years on their own shrinks the reference sample
(noisier PCA/cluster boundaries) and leaves too few consecutive-year pairs to
compute a meaningful transition matrix. For the live demo, upload a file that
appends the new/recent years to the full historical panel (same sheet, same
columns, new rows below the existing ones) rather than the new years alone.

## What's new (supervisor-facing features)

Added after reviewing the dashboard from the perspective of its intended
user — a bank-system risk supervision unit uploading fresh data, reading
results, and forming a view of sector health:

- **Data-context bar** right under the title: always shows how many banks and
  which year range are currently loaded, and which file (default vs
  uploaded). If banks' "latest year on record" isn't the same across the
  portfolio (some report later than others), a warning banner lists the
  breakdown by year so the Portfolio table isn't misread as one single
  snapshot date.
- **"Về phương pháp luận" expander**: a collapsible in-app summary of the
  CAMEL → PCA → K-means → transitions pipeline and its limitations (moved
  out of this README so an end user reading the live app — not the repo —
  still sees it).
- **System-wide trend chart** (Portfolio Overview tab): a stacked bar of bank
  counts per risk state, by year — shows whether the *whole system* has been
  getting worse or better over time, not just each bank's own latest state.
- **Raw CAMEL ratio table** (Bank Detail tab): the actual % ratios (ROE, NPL,
  CIR, etc.) for the selected bank/year, next to the existing standardized
  z-score chart — for reading against real-world thresholds, not just
  relative deviation.
- **NPL benchmark chart** (Bank Detail tab): the bank's NPL ratio over time
  with a reference line at 3%, the ceiling commonly cited in Vietnamese
  banking regulation as a safe-operation threshold. This line is a visual
  reference only, not an input to the clustering model.

## Deploy for the live demo

Push this folder to a GitHub repo and deploy free at
https://streamlit.io/cloud (Community Cloud) to get a shareable public link
for the presentation, or simply run it locally and share your screen.

## Methodology summary

1. Clean raw data, drop the policy bank (VBSP), compute the 11 CAMEL
   indicators (2 of them — NIE, PCR — are derived; the other 9 exist directly
   in the source dataset).
2. Risk-orient every indicator (flip sign for "higher = safer" ratios),
   apply log-modulus transform, standardize (z-score).
3. Run PCA within each of the 5 CAMEL blocks (Earnings, Management, Asset
   Quality, Liquidity, Capital), keep PC1 as that block's factor score,
   sign-orient it so higher = more risk.
4. Run K-means (k=5) on the 5-dimensional factor space, order clusters by
   aggregate risk into Low risk / Moderate / Watchlist / Stressed / Distress.
5. Compute one-year transition probabilities and per-bank timelines to
   surface early-warning signals (a bank drifting toward a worse state),
   not just a static snapshot.

Validated against the published paper's own results: our Watchlist/Stressed/
Distress cluster sizes (24/252/126) closely match theirs (23/246/123), and
our transition-matrix persistence ordering (Stressed stickiest, Watchlist
least sticky) matches their dwell-time findings — see the "Limitations" note
in the presentation for the one meaningful discrepancy (Low risk vs Moderate
boundary), which stems from re-deriving NIE and PCR from the paper's variable
descriptions rather than having their exact original formulas.
