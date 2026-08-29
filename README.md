# Bank Risk Early-Warning Supervision Dashboard (demo)

Reimplements the CAMEL block-wise PCA + K-means clustering methodology from
Chung, N.H. & Hung, V.T. (2026), "Bank risk clustering and early warning
supervision," *International Review of Economics and Finance*, 110, 105571.

## Files

- `pipeline.py` — data cleaning, CAMEL ratio construction, risk-orientation +
  log-modulus transform + standardization, block-wise PCA, K-means clustering,
  transition-matrix / timeline / alert computation. All reusable functions.
- `app.py` — Streamlit dashboard (3 pages: Portfolio Overview, Bank Detail,
  System Transitions, selected via a top nav instead of native tabs so a
  "View →" button can jump straight to a bank's detail page). Imports
  `pipeline.py`. All UI text is in English.
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

The app has a **file uploader in the sidebar** ("Upload updated data") — no
code edits or redeploy needed. Upload a new `.xlsx` file with the same
column layout as the Le et al. (2022) Vietnamese banking dataset (Bank Code,
Year, plus the raw balance-sheet/income-statement fields the pipeline maps to
the 11 CAMEL indicators), confirm the sheet name (default `Data`), and the
whole dashboard — clustering, transitions, alerts — re-fits on that file
live. Since PCA + K-means aren't a pretrained model but a fit-on-whatever-
data-you-give-it procedure, this "retrains" correctly every time; there is no
separate training step. Use the "Reset to default data" button to switch
back to the original 2002-2021 dataset. This is the intended way to demo a
live "test set" during the presentation.

If a file fails to load (wrong sheet name, missing/renamed columns), the app
shows a plain-English explanation (e.g. "Could not find a sheet named
'Data' in this file") and falls back to the default dataset rather than
crashing or showing a raw Python error. The underlying exception is still
available in a collapsed "Technical details" expander for debugging.

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

- **System-wide trend chart** (Portfolio Overview page): a stacked bar of
  bank counts per risk state, by year — shows whether the *whole system* has
  been getting worse or better over time, not just each bank's own latest
  state.
- **Raw CAMEL ratio table** (Bank Detail page): the actual % ratios (ROE,
  NPL, CIR, etc.) for the selected bank/year, next to the existing
  standardized z-score chart — for reading against real-world thresholds,
  not just relative deviation.
- **NPL benchmark chart** (Bank Detail page): the bank's NPL ratio over time
  with a reference line at 3%, the ceiling commonly cited in Vietnamese
  banking regulation as a safe-operation threshold. This line is a visual
  reference only, not an input to the clustering model. *(Removed in the
  "Latest update" round below — see that section.)*

## UX pass (5 fixes)

A second round of changes after a UI/UX review of the app itself:

1. **Less preamble before real content.** Removed the always-on data-context
   banner, the "mixed reporting years" warning, and the in-app methodology
   expander that used to sit between the title and the first page — a
   returning user (e.g. daily/weekly use) now reaches the actual data
   immediately. (Methodology detail lives in this README's "Methodology
   summary" section instead.)
2. **Color isn't the only signal for risk state.** Every risk-state label —
   KPI cards, portfolio table, alerts, chart legends and axes — is now
   paired with a distinct shape icon (`●` Low risk, `◆` Moderate, `▲`
   Watchlist, `■` Stressed, `✖` Distress), so the 5 states stay
   distinguishable for red-green color-blind viewers, not just by hue.
3. **Fixed a real clipping bug**: on the CAMEL factor breakdown chart, a
   large deviation value's outside label could get cut off at the plot edge
   (e.g. "-0.44" rendering as ").44"). Fixed with `cliponaxis=False`,
   auto-margins, and a padded x-axis range.
4. **Drill-down from alerts to bank detail.** Each row in "Recent alerts"
   now has a "View →" button that jumps straight to the Bank Detail page
   with that bank pre-selected, instead of requiring the user to remember
   the bank code and re-select it from a dropdown. (Implemented by
   replacing `st.tabs` with a session-state-backed radio nav, since
   Streamlit's native tabs can't be switched programmatically.) *(The
   "Recent alerts" widget itself was later removed — see "Latest update"
   below — but the session-state radio nav it required stays in place,
   since Bank Detail's own bank selector still benefits from it.)*
5. **Friendlier upload errors.** A bad upload (wrong sheet name, missing
   columns) now shows a plain-English message telling the user what to
   check, instead of a raw Python exception; the original exception is
   still available in a collapsed "Technical details" expander.

## Latest update (dashboard adjustments)

A further round of changes, based on direct review of the dashboard:

**Portfolio Overview**
- Removed the "Recent alerts" widget.
- The Portfolio table now has three filters (risk group, year, bank) and
  pulls from the full multi-year history instead of just each bank's latest
  year — defaults to the most recent year, but older years or specific
  banks can be added.

**Bank Detail**
- Added a year selector that drives both the "CAMEL factor breakdown" chart
  and the "Actual CAMEL ratios (%)" table together.
- Rewrote "Suggested areas to investigate" to be clearer and more
  action-oriented business English (e.g. "Dig into the NIM trend...",
  "Request a provisioning-adequacy review if coverage looks thin...").
- Removed the "NPL ratio over time" widget.
- "Actual CAMEL ratios (%)" table now spans the full row width, with a
  taller row height so all 11 indicators show without scrolling. A caption
  below the table flags that this table shows the single selected year's
  raw figures, while the CAMEL factor breakdown chart above benchmarks
  against the full 2002-2021 panel average — the two can point in
  different directions for the same bank-year, which reflects two
  different reference points rather than a bug.

**System Transitions**
- Added a "Filter by bank" selector. The system-wide transition matrix and
  persistence chart stay aggregate (that is their structural basis, drawn
  from `transition_probs`), and selecting a specific bank adds a table
  below — built from the per-bank `timelines` data — showing that bank's
  own year-by-year transitions, with each move flagged Worsened / Improved
  / No change.

A peer-average benchmark column (comparing each ratio in "Actual CAMEL
ratios" against the same-year average across all banks) was prototyped and
reviewed, but not adopted for this version — the team decided the added
complexity (and the question of whether a data-driven peer average or an
official regulatory threshold is the more defensible benchmark) needed
more discussion before shipping it.

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
