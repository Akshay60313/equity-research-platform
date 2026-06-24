# 📊 Equity Research & Valuation Platform

A live DCF (Discounted Cash Flow) valuation tool built in Python and Streamlit.
Enter any public company ticker and get a full equity research workup — live
financial data, an automated 5-year DCF valuation, sensitivity analysis,
scenario modelling, peer comparison, and a business quality score.

**🔗 Live Demo:** https://equity-research-platform-1.streamlit.app



<!-- 
  Replace the line above with an embedded video once recorded, e.g.:
  [![Watch the demo](thumbnail.png)](your-video-link-here)
-->

---

## What it does

- **Live data** — pulls real financials directly from Yahoo Finance
- **DCF Valuation** — full 5-year free cash flow build with proper tax treatment,
  stock-based compensation addback, and terminal value via Gordon Growth
- **Sensitivity Analysis** — WACC vs Terminal Growth heatmap, built the same way
  a standard Excel sensitivity table works
- **Reverse DCF** — backs out what growth rate the market is currently pricing in
- **Scenario Analysis** — Bear/Base/Bull cases, all using the same full 5-year
  DCF mechanism for consistency
- **5-Year Return Model** — exit multiple based projected return and CAGR
- **Peer Comparison** — P/S, P/E, EV/Revenue, EV/EBITDA across sector peers
- **Business Quality Score** — weighted scoring across growth, profitability,
  capital efficiency, leverage, and valuation
- **Data validation** — cross-checks share count against an independently
  derived figure, and clamps unrealistic growth rates before they distort the model

## Validated against real models

The core DCF mechanism was built and stress-tested against independently
researched Excel models (PepsiCo, Snowflake) — not just assumed correct.
**Works well for mature, profitable, stable-margin companies** (validated to
within ~6% of intrinsic value). See the in-app **Limitations tab** for an
honest breakdown of what this tool handles well and what it doesn't
(extreme-SBC companies, hyper-growth names, banks, REITs, commodities).

## Tech stack

Python · Streamlit · yfinance · pandas · NumPy · Plotly

## Run locally

```bash
pip install streamlit yfinance pandas plotly numpy
streamlit run app.py
```

## Key engineering decisions

- **FCF formula**: `EBIT × (1 - tax) + D&A + SBC - Capex - ΔWorking Capital`
  — standard FCFF, with SBC capped so it cannot fully erase a real operating loss
- **Growth/margin taper**: Year 1 (live data, clamped at 35%) tapers linearly
  to a user-set Year 5 "normalized" assumption — avoids holding a temporary
  growth spike flat for 5 years
- **Scenarios use the full 5-year DCF**, not a shortcut — guarantees Bull ≥
  Base ≥ Bear and keeps every tab internally consistent
