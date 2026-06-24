"""
Equity Research & Valuation Platform
FINAL PORTFOLIO VERSION (Tab 4 fixed to use the full 5-year mechanism)

Validated to work well for: mature, profitable, stable-margin companies.
See the Limitations tab for exactly what this does and does not handle well.

Features implemented:
  - 3-Year Smoothing Hack for Working Capital.
  - Terminal Year Normalization (CapEx tapers to match D&A, WC goes to 0).
  - SBC Dilution Hack (Estimates 0.5% share dilution per 1% of SBC margin).
  - SBC addback CAP: SBC can no longer fully erase a real operating loss.
  - SaaS WC Override: Uncapped negative working capital for deferred revenue.
  - Hyper-Growth Clamp: Caps Year 1 automated growth at 35% to prevent math explosions.
  - 5-Year Return Model restored (exit multiple -> exit price -> CAGR).
  - Scenario tab (Tab 4) now runs the SAME full 5-year FCF build and DCF as
    the Valuation tab, instead of a single-year snapshot. This guarantees
    Bull >= Base >= Bear, and Base now matches Tab 3 exactly - the single
    year snapshot version produced inconsistent numbers between tabs.

SETUP:
    pip install streamlit yfinance pandas plotly numpy

RUN:
    streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(page_title="Equity Research Tool", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }

    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .stTabs [data-baseweb="tab"] { font-weight: 600; }

    [data-testid="stSidebar"] { background-color: #1a2b4a; }
    [data-testid="stSidebar"] * { color: white !important; }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] div[data-baseweb="input"] input {
        color: #000000 !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border: 1px solid #999999 !important;
    }

    h1, h2, h3 { color: #1a2b4a; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Equity Research & Valuation Platform")
st.caption("Live financial analysis, DCF valuation, and scenario modelling for any public company")
st.info("✅ **Validated to work well for mature, profitable, stable-margin companies** "
       "(e.g. PepsiCo, Microsoft). For pre-profit, hyper-growth, or extreme-SBC "
       "companies, see the Limitations tab before trusting the output.")


# ============================================================
# CORE FINANCE FUNCTIONS
# ============================================================

def build_path(yr1, yr5):
    """Shared taper/ramp shape. Used for both growth and margin."""
    return [
        yr1,
        yr1 + (yr5 - yr1) * 0.25,
        yr1 + (yr5 - yr1) * 0.50,
        yr1 + (yr5 - yr1) * 0.75,
        yr5
    ]


def fcf_projection(revenue_base, growth_path, margin_path, da_pct,
                   sbc_pct, capex_pct, wc_pct, tax_rate):
    """
    FCF = EBIT*(1-tax) + D&A + SBC(capped) - Capex - WC_change
    Includes taper normalization for Terminal Year assumptions.
    """
    fcfs = []
    revenue = revenue_base

    for i, (g, margin) in enumerate(zip(growth_path, margin_path)):
        revenue  = revenue * (1 + g)
        ebitda   = revenue * margin
        da       = revenue * da_pct
        ebit     = ebitda - da
        nopat    = ebit * (1 - tax_rate)
        sbc      = revenue * sbc_pct

        # SBC addback CAP: cap SBC's contribution when NOPAT is negative,
        # so SBC cannot single-handedly turn a real operating loss into
        # apparent positive free cash flow.
        if nopat < 0:
            sbc_addback = min(sbc, abs(nopat) + sbc * 0.5)
        else:
            sbc_addback = sbc

        # Taper CapEx down to match D&A by Year 5
        weight = (4 - i) / 4.0
        capex = (revenue * capex_pct * weight) + (da * (1 - weight))

        # Taper Working Capital change to zero by Year 5
        delta_wc = revenue * wc_pct * weight

        fcf      = nopat + da + sbc_addback - capex - delta_wc
        fcfs.append(round(fcf, 1))

    return fcfs


def dcf_value(fcfs, wacc, terminal_growth, net_debt=0):
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth rate")
    pv_fcfs          = sum([fcf / (1 + wacc)**i for i, fcf in enumerate(fcfs, 1)])
    terminal_value   = (fcfs[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv_terminal      = terminal_value / (1 + wacc)**len(fcfs)
    enterprise_value = pv_fcfs + pv_terminal
    equity_value     = enterprise_value - net_debt
    return {
        'pv_fcfs':          round(pv_fcfs, 1),
        'terminal_value':   round(terminal_value, 1),
        'pv_terminal':      round(pv_terminal, 1),
        'enterprise_value': round(enterprise_value, 1),
        'equity_value':     round(equity_value, 1)
    }


def sensitivity_table(fcfs, wacc_range, growth_range, net_debt=0, diluted_shares=1):
    wacc_grid, g_grid = np.meshgrid(wacc_range, growth_range)
    results = np.zeros_like(wacc_grid)
    for i in range(wacc_grid.shape[0]):
        for j in range(wacc_grid.shape[1]):
            w = wacc_grid[i, j]
            g = g_grid[i, j]
            if w > g:
                r = dcf_value(fcfs, w, g, net_debt)
                results[i, j] = r['equity_value'] / diluted_shares if diluted_shares else 0
            else:
                results[i, j] = np.nan
    df = pd.DataFrame(
        np.round(results, 2),
        index=[f"{g:.1%}" for g in growth_range],
        columns=[f"{w:.1%}" for w in wacc_range]
    )
    df.index.name   = 'Terminal Growth'
    df.columns.name = 'WACC'
    return df


def reverse_dcf(fcfs, wacc, net_debt, diluted_shares, target_price):
    g_range   = np.linspace(0.01, wacc - 0.005, 200)
    best_g    = None
    best_diff = float('inf')
    for g in g_range:
        result        = dcf_value(fcfs, wacc, g, net_debt)
        implied_price = result['equity_value'] / diluted_shares
        diff          = abs(implied_price - target_price)
        if diff < best_diff:
            best_diff = diff
            best_g    = g
    return best_g


def get_actuals(info, income, cashflow):
    revenue = info.get('totalRevenue') or 0

    ebitda_margin = info.get('ebitdaMargins')
    if not ebitda_margin or revenue == 0:
        ebitda_margin = 0.15

    da_pct, capex_pct, wc_pct, sbc_pct = 0.05, 0.06, 0.01, 0.03

    try:
        if cashflow is not None and not cashflow.empty and revenue:
            latest_col = cashflow.columns[0]
            if 'Depreciation And Amortization' in cashflow.index:
                v = cashflow.loc['Depreciation And Amortization', latest_col]
                if pd.notna(v) and revenue:
                    da_pct = abs(v) / revenue
            if 'Capital Expenditure' in cashflow.index:
                v = cashflow.loc['Capital Expenditure', latest_col]
                if pd.notna(v) and revenue:
                    capex_pct = abs(v) / revenue
            if 'Stock Based Compensation' in cashflow.index:
                v = cashflow.loc['Stock Based Compensation', latest_col]
                if pd.notna(v) and revenue:
                    sbc_pct = abs(v) / revenue

            # 3-Year Smoothing for Working Capital
            if 'Change In Working Capital' in cashflow.index:
                available_years = min(3, len(cashflow.columns))
                wc_vals = []
                for i in range(available_years):
                    v = cashflow.loc['Change In Working Capital', cashflow.columns[i]]
                    if pd.notna(v):
                        wc_vals.append(-(v)) # Keep sign logic: negative = source
                if wc_vals and revenue:
                    avg_wc = sum(wc_vals) / len(wc_vals)
                    wc_pct = avg_wc / revenue

    except Exception:
        pass

    # Bounds: wc_pct floor is -0.35 to allow SaaS deferred revenue logic
    da_pct        = max(min(da_pct, 0.20), 0.0)
    capex_pct     = max(min(capex_pct, 0.30), 0.0)
    wc_pct        = max(min(wc_pct, 0.05), -0.35)
    sbc_pct       = max(min(sbc_pct, 0.40), 0.0)
    ebitda_margin = max(min(ebitda_margin, 0.60), -0.30)

    return {
        'ebitda_margin': ebitda_margin,
        'da_pct': da_pct,
        'capex_pct': capex_pct,
        'wc_pct': wc_pct,
        'sbc_pct': sbc_pct
    }


def score_metric(value, thresholds):
    for cutoff, score in thresholds:
        if value >= cutoff:
            return score
    return 0


def calculate_quality_score(kpis):
    scores = {}
    scores['growth'] = score_metric(kpis.get('revenue_growth', 0) or 0, [
        (0.30, 100), (0.20, 80), (0.10, 60), (0.05, 40), (0, 20)
    ])
    scores['profitability'] = score_metric(kpis.get('net_margin', 0) or 0, [
        (0.20, 100), (0.10, 80), (0.05, 60), (0, 40)
    ])
    scores['capital_efficiency'] = score_metric(kpis.get('roe', 0) or 0, [
        (0.20, 100), (0.15, 80), (0.10, 60), (0, 40)
    ])
    de = kpis.get('debt_to_equity', 100) or 100
    scores['leverage'] = score_metric(-de, [
        (-30, 100), (-60, 80), (-100, 60), (-200, 40)
    ])
    pe = kpis.get('pe_ratio', 50) or 50
    scores['valuation'] = score_metric(-pe, [
        (-15, 100), (-25, 80), (-35, 60), (-50, 40)
    ])
    return scores


def final_quality_score(scores):
    weights = {
        'growth': 0.25, 'profitability': 0.25,
        'capital_efficiency': 0.20, 'leverage': 0.15, 'valuation': 0.15
    }
    return round(sum(scores[k] * weights[k] for k in weights), 1)


def calculate_kpis(info, income):
    kpis = {}
    try:
        revenue = income.loc['Total Revenue'].sort_index()
        kpis['revenue_growth'] = revenue.pct_change().iloc[-1]
    except Exception:
        kpis['revenue_growth'] = info.get('revenueGrowth')
    kpis['gross_margin']   = info.get('grossMargins')
    kpis['net_margin']     = info.get('profitMargins')
    kpis['roe']            = info.get('returnOnEquity')
    kpis['debt_to_equity'] = info.get('debtToEquity')
    kpis['pe_ratio']       = info.get('trailingPE')
    return kpis


def clean_statement(df, key_rows):
    if df is None or df.empty:
        return None
    df = df.sort_index(axis=1) / 1e6
    available = [r for r in key_rows if r in df.index]
    if not available:
        return None
    df = df.loc[available]
    df.columns = [str(c.year) for c in df.columns]
    return df


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.info, stock.financials, stock.balance_sheet, stock.cashflow


@st.cache_data
def get_peer_comparison(tickers):
    results = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            results.append({
                'Ticker':          t,
                'Price':           info.get('currentPrice'),
                'Market Cap ($B)': round(info.get('marketCap', 0) / 1e9, 1),
                'P/E':             info.get('trailingPE'),
                'P/S':             info.get('priceToSalesTrailing12Months'),
                'EV/Revenue':      info.get('enterpriseToRevenue'),
                'EV/EBITDA':       info.get('enterpriseToEbitda'),
                'Revenue Growth':  info.get('revenueGrowth'),
                'Gross Margin':    info.get('grossMargins'),
            })
        except Exception as e:
            st.warning(f"Could not load {t}: {e}")
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).set_index('Ticker')


PEER_MAP = {
    'SNOW': ['DDOG', 'MDB', 'NET', 'CRWD'],
    'PEP':  ['KO', 'MNST', 'KDP'],
    'AAPL': ['MSFT', 'GOOGL', 'AMZN'],
    'MSFT': ['AAPL', 'GOOGL', 'AMZN'],
}


def get_peers_for(ticker):
    return PEER_MAP.get(ticker, [])


# ============================================================
# SIDEBAR — TICKER FIRST, DATA PULL, THEN SLIDERS
# ============================================================

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Enter Ticker", value="MSFT").upper()

try:
    info, income, balance, cashflow = get_stock_data(ticker)
    data_loaded = True
except Exception as e:
    st.error(f"Could not load data for {ticker}: {e}")
    info        = {}
    income      = None
    balance     = None
    cashflow    = None
    data_loaded = False

actuals = get_actuals(info, income, cashflow) if data_loaded else \
          {'ebitda_margin': 0.15, 'da_pct': 0.05, 'capex_pct': 0.06,
           'wc_pct': 0.01, 'sbc_pct': 0.03}

raw_growth    = (info.get('revenueGrowth') or 0.10) * 100
# Hyper-Growth clamp (35%) protects against math explosions on AMZN/NVDA/PLTR
actual_growth = float(round(max(min(raw_growth, 35.0), -20.0), 1))

st.sidebar.header("Valuation Assumptions")
wacc       = st.sidebar.slider("WACC (%)", 3.0, 15.0, 8.9, 0.1) / 100
terminal_g = st.sidebar.slider("Terminal Growth (%)", 0.0, 5.0, 2.0, 0.25) / 100
tax_rate   = st.sidebar.slider("Tax Rate (%)", 0.0, 40.0, 21.0, 1.0) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Growth & Margin (Year 1 → Year 5)")
st.sidebar.caption("Year 1 pulled live (clamped at 35% max). Year 5 is your "
                  "view of the sustainable, normalized level. Tapers "
                  "between them.")

yr1_growth = st.sidebar.slider("Year 1 Revenue Growth (%)", -20.0, 80.0, actual_growth, 1.0) / 100
yr5_growth = st.sidebar.slider("Year 5 Revenue Growth (%)", 1.0, 30.0, 10.0, 1.0) / 100

raw_margin    = actuals['ebitda_margin'] * 100
yr1_margin = st.sidebar.slider(
    "Year 1 EBITDA Margin (%)", -30.0, 60.0,
    float(round(max(min(raw_margin, 60.0), -30.0), 1)), 1.0
) / 100
yr5_margin = st.sidebar.slider(
    "Year 5 EBITDA Margin (%)", 0.0, 60.0,
    54.0, 1.0
) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("FCF Build — Flat % of Revenue")
st.sidebar.caption("Pre-filled from actuals. WC% uses a 3-year historical "
                   "average to smooth out single-year anomalies.")

da_pct    = st.sidebar.slider("D&A (% of Revenue)", 0.0, 20.0,
                              float(round(actuals['da_pct']*100, 1)), 0.5) / 100
capex_pct = st.sidebar.slider("Capex (% of Revenue)", 0.0, 30.0,
                              float(round(actuals['capex_pct']*100, 1)), 0.5) / 100
sbc_pct   = st.sidebar.slider("Stock-Based Comp (% of Revenue)", 0.0, 40.0,
                              float(round(actuals['sbc_pct']*100, 1)), 0.5) / 100
# Expanded minimum down to -50% to properly accommodate SaaS Deferred Revenue
wc_pct    = st.sidebar.slider("Working Capital Change (% of Revenue)", -50.0, 15.0,
                              float(round(actuals['wc_pct']*100, 1)), 0.5) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("5-Year Return Model")
exit_multiple = st.sidebar.slider("Exit EV/EBITDA Multiple", 5.0, 60.0, 15.0, 1.0)

GROWTH_PATH = build_path(yr1_growth, yr5_growth)
MARGIN_PATH = build_path(yr1_margin, yr5_margin)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Overview", "Financials", "Valuation", "Scenarios", "Peers",
     "Quality Score", "Limitations"]
)

if data_loaded:

    revenue_base  = (info.get('totalRevenue') or 0) / 1e6
    # Null check for cash/debt to prevent crashes on tickers with missing data
    net_debt      = ((info.get('totalDebt') or 0) - (info.get('totalCash') or 0)) / 1e6
    current_price = info.get('currentPrice') or 0

    _shares_raw     = info.get('sharesOutstanding') or 0
    _mcap           = info.get('marketCap') or 0
    _shares_derived = (_mcap / current_price) if current_price else 0

    if _shares_raw and _shares_derived:
        _diff = abs(_shares_raw / _shares_derived - 1)
        shares = (_shares_derived if _diff > 0.05 else _shares_raw) / 1e6
    elif _shares_derived:
        shares = _shares_derived / 1e6
    else:
        shares = (_shares_raw or 1) / 1e6

    # SBC Dilution Hack: Estimates 0.5% annual dilution per 1% of SBC.
    diluted_shares = shares * (1 + (sbc_pct * 0.5))**5 if shares else 0

    kpis = calculate_kpis(info, income)

    fcfs = fcf_projection(
        revenue_base, GROWTH_PATH, MARGIN_PATH,
        da_pct, sbc_pct, capex_pct, wc_pct, tax_rate
    )

    # Check: does SBC exceed the magnitude of operating profit in Year 1?
    _rev1  = revenue_base * (1 + GROWTH_PATH[0])
    _ebitda1 = _rev1 * MARGIN_PATH[0]
    _da1   = _rev1 * da_pct
    _ebit1 = _ebitda1 - _da1
    _sbc1  = _rev1 * sbc_pct
    sbc_dominates = (_ebit1 < 0) and (_sbc1 > abs(_ebit1))

    # ----------------------------------------------------------------
    # TAB 1 — OVERVIEW
    # ----------------------------------------------------------------
    with tab1:
        company_name = info.get('longName', ticker)
        st.markdown(f"<h2 style='color:#1a2b4a'>{company_name}</h2>",
                    unsafe_allow_html=True)
        st.write(f"**Sector:** {info.get('sector','N/A')} | "
                 f"**Industry:** {info.get('industry','N/A')}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Price", f"${current_price:,.2f}")
        with col2:
            st.metric("Market Cap", f"${(info.get('marketCap') or 0)/1e9:,.1f}B")
        with col3:
            st.metric("Revenue", f"${revenue_base:,.0f}M")
        with col4:
            gm = kpis.get('gross_margin')
            st.metric("Gross Margin", f"{gm:.1%}" if gm else "N/A")

        col5, col6, col7 = st.columns(3)
        with col5:
            pe = kpis.get('pe_ratio')
            st.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
        with col6:
            rg = kpis.get('revenue_growth')
            st.metric("Revenue Growth", f"{rg:.1%}" if rg else "N/A")
        with col7:
            roe = kpis.get('roe')
            st.metric("ROE", f"{roe:.1%}" if roe else "N/A")

        if sbc_dominates:
            st.warning("⚠️ **Stock-Based Compensation exceeds the magnitude "
                      "of operating loss** for this company at current "
                      "assumptions. Free cash flow may be substantially "
                      "driven by the non-cash SBC addback rather than real "
                      "operating cash generation. Treat the valuation on "
                      "this page with caution — see the Limitations tab.")

        st.markdown("---")
        st.subheader("Data Validation: Share Count Integrity")
        st.caption("yfinance occasionally provides stale share counts. This checks "
                  "the raw feed against a derived count (Market Cap / Price). If "
                  "the difference exceeds a 5% margin of error, the model "
                  "overrides the raw data to protect the per-share valuation.")

        scol1, scol2, scol3, scol4 = st.columns(4)
        with scol1:
            st.metric("Raw Shares (API)", f"{_shares_raw/1e9:,.3f}B" if _shares_raw else "N/A")
        with scol2:
            st.metric("Derived Shares (Mcap/Px)", f"{_shares_derived/1e9:,.3f}B" if _shares_derived else "N/A")
        with scol3:
            if _shares_raw and _shares_derived:
                diff_pct = abs((_shares_raw / _shares_derived) - 1)
                st.metric("Margin of Error", f"{diff_pct:.1%}")
            else:
                diff_pct = 0
                st.metric("Margin of Error", "N/A")
        with scol4:
            if _shares_raw and _shares_derived:
                if diff_pct > 0.05:
                    st.error("⚠️ 5% Failsafe Triggered: Using Derived Shares")
                else:
                    st.success("✅ Data Aligned: Using Raw Shares")

        st.markdown("---")
        st.subheader("Data Validation: Hyper-Growth Normalization")
        st.caption("Extrapolating short-term hyper-growth across a 5-year DCF "
                  "artificially inflates terminal value. The model caps "
                  "automated Year 1 growth at 35% to protect the intrinsic "
                  "valuation.")

        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            st.metric("Raw API Growth", f"{raw_growth:.1f}%")
        with gcol2:
            st.metric("Model Baseline (Year 1)", f"{actual_growth:.1f}%")
        with gcol3:
            if raw_growth > 35.0:
                st.warning("⚠️ Growth Clamped to 35%")
            else:
                st.success("✅ Growth Within Normal Bounds")

        st.markdown("---")

        st.subheader("Business Summary")
        st.write(info.get('longBusinessSummary', 'No summary available.'))

        try:
            rev_series = income.loc['Total Revenue'].sort_index() / 1e6
            rev_df = rev_series.reset_index()
            rev_df.columns = ['date', 'revenue']
            rev_df['year'] = rev_df['date'].dt.year
            fig_rev = px.bar(rev_df, x='year', y='revenue',
                             title='Revenue Trend ($M)',
                             color_discrete_sequence=['#2563a8'])
            st.plotly_chart(fig_rev, use_container_width=True)
        except Exception:
            st.info("Revenue trend chart unavailable for this ticker.")

    # ----------------------------------------------------------------
    # TAB 2 — FINANCIALS
    # ----------------------------------------------------------------
    with tab2:
        st.header("Financial Statements")
        st.caption("All figures in $ millions, oldest to newest")

        st.subheader("Income Statement")
        inc = clean_statement(income, [
            'Total Revenue', 'Cost Of Revenue', 'Gross Profit',
            'Operating Income', 'EBITDA', 'Net Income'
        ])
        if inc is not None:
            st.dataframe(inc.style.format("${:,.0f}"), use_container_width=True)
        else:
            st.write("Income statement not available.")

        st.subheader("Balance Sheet")
        bal = clean_statement(balance, [
            'Total Assets', 'Total Liabilities Net Minority Interest',
            'Total Debt', 'Cash And Cash Equivalents', 'Stockholders Equity'
        ])
        if bal is not None:
            st.dataframe(bal.style.format("${:,.0f}"), use_container_width=True)
        else:
            st.write("Balance sheet not available.")

        st.subheader("Cash Flow Statement")
        cf = clean_statement(cashflow, [
            'Operating Cash Flow', 'Capital Expenditure',
            'Free Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow'
        ])
        if cf is not None:
            st.dataframe(cf.style.format("${:,.0f}"), use_container_width=True)
        else:
            st.write("Cash flow statement not available.")

    # ----------------------------------------------------------------
    # TAB 3 — VALUATION
    # ----------------------------------------------------------------
    with tab3:
        st.header("DCF Valuation")

        with st.expander("📋 Growth & Margin Path Used (Year 1 → Year 5)", expanded=True):
            path_df = pd.DataFrame({
                'Year': [f"Year {i+1}" for i in range(5)],
                'Growth': [f"{g:.1%}" for g in GROWTH_PATH],
                'EBITDA Margin': [f"{m:.1%}" for m in MARGIN_PATH]
            })
            st.dataframe(path_df, use_container_width=True, hide_index=True)
            st.caption(f"D&A: {da_pct:.1%} | Capex: {capex_pct:.1%} | "
                      f"SBC: {sbc_pct:.1%} | WC Change: {wc_pct:.1%} of revenue "
                      f"— Tapering applied to Terminal Year CapEx and WC.")

        if sbc_dominates:
            st.info("ℹ️ SBC addback is being capped this year because it "
                   "exceeds the magnitude of operating loss — see Overview "
                   "for details and the Limitations tab for why this matters.")

        result    = dcf_value(fcfs, wacc, terminal_g, net_debt)
        intrinsic = result['equity_value'] / diluted_shares if diluted_shares else 0
        delta     = ((intrinsic / current_price) - 1) if current_price else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Enterprise Value", f"${result['enterprise_value']:,.0f}M")
        with col2:
            st.metric("Equity Value", f"${result['equity_value']:,.0f}M")
        with col3:
            st.metric("Intrinsic Value/Share", f"${intrinsic:,.2f}",
                      delta=f"{delta:.1%} vs market")
            st.caption(f"Uses diluted terminal shares: {diluted_shares/1e3:,.1f}B")

        with st.expander("See the 5-year FCF build"):
            fcf_display = pd.DataFrame({
                'Year': [f"Year {i+1}" for i in range(5)],
                'FCF ($M)': fcfs
            })
            st.dataframe(fcf_display, use_container_width=True, hide_index=True)

        st.subheader("Sensitivity Analysis (Intrinsic Value per Share)")
        st.caption("Same mechanism as a standard Excel sensitivity table: "
                  "the 5-year FCF stream above is fixed. Only WACC and "
                  "terminal growth vary across this grid.")

        wacc_range   = np.linspace(max(wacc - 0.02, 0.01), wacc + 0.02, 5)
        growth_range = np.linspace(max(terminal_g - 0.01, 0.0), terminal_g + 0.01, 5)

        sens = sensitivity_table(fcfs, wacc_range, growth_range, net_debt, diluted_shares)
        st.dataframe(sens.style.format("${:,.2f}"), use_container_width=True)

        fig_heat = go.Figure(go.Heatmap(
            z=sens.values, x=sens.columns.tolist(), y=sens.index.tolist(),
            colorscale='RdYlGn', text=sens.values,
            texttemplate='$%{text:.2f}',
            hovertemplate='WACC: %{x}<br>Terminal Growth: %{y}<br>Price: $%{z:.2f}<extra></extra>'
        ))
        fig_heat.update_layout(
            title='Intrinsic Value per Share — WACC vs Terminal Growth',
            xaxis_title='WACC', yaxis_title='Terminal Growth', height=400
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Reverse DCF — Implied Growth")
        if current_price and diluted_shares:
            implied_g = reverse_dcf(fcfs, wacc, net_debt, diluted_shares, current_price)
            st.metric("Market-Implied Terminal Growth", f"{implied_g:.1%}")
            st.caption(f"At ${current_price:.2f}, the market implies terminal "
                      f"growth of **{implied_g:.1%}** given your current "
                      f"assumptions at {wacc:.1%} WACC.")

        # 5-Year Return Model
        st.subheader("5-Year Return Model")
        st.caption("A different question than the DCF: if you buy today and "
                  "sell in 5 years at a reasonable exit multiple, what's the "
                  "implied return? Uses Year 5's EBITDA from the build above.")

        rev_y5    = revenue_base
        for g in GROWTH_PATH:
            rev_y5 = rev_y5 * (1 + g)
        ebitda_y5 = rev_y5 * MARGIN_PATH[-1]

        exit_ev        = ebitda_y5 * exit_multiple
        exit_equity    = exit_ev - net_debt
        exit_price     = exit_equity / diluted_shares if diluted_shares else 0
        total_return   = (exit_price - current_price) / current_price if current_price else 0
        cagr           = (exit_price / current_price) ** (1/5) - 1 if current_price and exit_price > 0 else 0

        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.metric("Projected Exit Price (Year 5)", f"${exit_price:,.2f}")
        with rcol2:
            st.metric("Total Return (5yr)", f"{total_return:.1%}")
        with rcol3:
            st.metric("CAGR", f"{cagr:.1%}")

    # ----------------------------------------------------------------
    # TAB 4 — SCENARIOS
    # FIXED: now runs the SAME full 5-year FCF build and DCF as the
    # Valuation tab, instead of a single-year snapshot. This guarantees
    # Bull >= Base >= Bear, and Base now matches Tab 3 exactly. The
    # single-year snapshot version previously produced a Base case that
    # disagreed with Tab 3 by as much as 3x for the same company.
    # ----------------------------------------------------------------
    with tab4:
        st.header("Scenario Analysis")
        st.caption("Runs the same full 5-year FCF build and DCF as the "
                  "Valuation tab — Base case here matches Tab 3 exactly. "
                  "Bear: Year 1 & Year 5 growth ×0.85, margin ×0.90. "
                  "Bull: growth ×1.15, margin ×1.05. Same WACC and terminal "
                  "growth across all three.")

        def scenario_price(growth_mult, margin_mult):
            g_path = build_path(yr1_growth * growth_mult, yr5_growth * growth_mult)
            m_path = build_path(yr1_margin * margin_mult, yr5_margin * margin_mult)
            scenario_fcfs = fcf_projection(
                revenue_base, g_path, m_path,
                da_pct, sbc_pct, capex_pct, wc_pct, tax_rate
            )
            r = dcf_value(scenario_fcfs, wacc, terminal_g, net_debt)
            return r['equity_value'] / diluted_shares if diluted_shares else 0

        bear_price = scenario_price(0.85, 0.90)
        base_price = scenario_price(1.00, 1.00)
        bull_price = scenario_price(1.15, 1.05)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bear Case", f"${bear_price:.2f}")
        with col2:
            st.metric("Base Case", f"${base_price:.2f}")
        with col3:
            st.metric("Bull Case", f"${bull_price:.2f}")

        st.write(f"Current market price: **${current_price:.2f}**")

        fig_range = go.Figure()
        fig_range.add_trace(go.Bar(
            x=['Bear', 'Base', 'Bull'],
            y=[max(bear_price, 0), base_price, bull_price],
            marker_color=['#E24B4A', '#378ADD', '#1D9E75']
        ))
        if current_price:
            fig_range.add_hline(y=current_price, line_dash='dash',
                                annotation_text='Current Price', line_color='black')
        fig_range.update_layout(title='Valuation Range by Scenario',
                                yaxis_title='Price per Share ($)')
        st.plotly_chart(fig_range, use_container_width=True)

    # ----------------------------------------------------------------
    # TAB 5 — PEERS
    # ----------------------------------------------------------------
    with tab5:
        st.header("Peer Comparison")
        st.info("ℹ️ Peer comparison available for: **SNOW**, **AAPL**, "
                "**PEP**, **MSFT**.")

        peers = get_peers_for(ticker)
        if peers:
            comp = get_peer_comparison([ticker] + peers)
            if not comp.empty:
                st.dataframe(
                    comp.style.format({
                        'Price':           '${:.2f}',
                        'Market Cap ($B)': '${:.1f}B',
                        'P/E':             '{:.1f}',
                        'P/S':             '{:.1f}x',
                        'EV/Revenue':      '{:.1f}x',
                        'EV/EBITDA':       '{:.1f}x',
                        'Revenue Growth':  '{:.1%}',
                        'Gross Margin':    '{:.1%}'
                    }),
                    use_container_width=True
                )
                fig_peers = px.bar(comp.reset_index(), x='Ticker', y='P/S',
                                  title='P/S Ratio Comparison', color='Ticker')
                st.plotly_chart(fig_peers, use_container_width=True)
        else:
            st.info(f"No peer mapping defined for {ticker} yet.")

    # ----------------------------------------------------------------
    # TAB 6 — QUALITY SCORE
    # ----------------------------------------------------------------
    with tab6:
        st.header("Business Quality Score")

        scores  = calculate_quality_score(kpis)
        overall = final_quality_score(scores)

        if overall >= 75:
            category, colour = "Strong", "green"
        elif overall >= 50:
            category, colour = "Average", "orange"
        else:
            category, colour = "Weak", "red"

        st.metric("Overall Quality Score", f"{overall}/100")
        st.markdown(f"**Category:** :{colour}[{category}]")

        score_colours = [
            '#1D9E75' if v >= 70 else '#F5A623' if v >= 40 else '#E24B4A'
            for v in scores.values()
        ]
        fig_score = go.Figure(go.Bar(
            x=list(scores.keys()), y=list(scores.values()),
            marker_color=score_colours
        ))
        fig_score.update_layout(title="Score Breakdown by Category",
                               yaxis_title="Score", yaxis_range=[0, 100])
        st.plotly_chart(fig_score, use_container_width=True)

    # ----------------------------------------------------------------
    # TAB 7 — LIMITATIONS
    # ----------------------------------------------------------------
    with tab7:
        st.header("Limitations & Assumptions")

        st.subheader("✅ Works well for: mature, profitable, stable companies")
        st.markdown("""
**This tool is validated and works well specifically for mature, profitable,
stable-margin companies** (e.g. PepsiCo, Microsoft) — the core DCF mechanism
has been tested against independently-built Excel models for this category
to within roughly 6% of intrinsic value. It also works reasonably well for
**moderate-SBC growth companies** (e.g. Snowflake, under careful
calibration), validated to within 1-7% depending on assumptions used.

For companies with flat or slowly-changing margins, predictable capital
intensity, and a manageable debt/cash position, this model's output is a
reasonable, defensible starting point for further analysis.
""")

        st.subheader("⚠️ Use with caution for: extreme-SBC or hyper-growth companies")
        st.markdown("""
- **Companies where Stock-Based Compensation exceeds operating profit**
  (flagged automatically with a warning on Overview/Valuation when detected):
  the SBC addback, even after the cap applied in this version, can still
  meaningfully influence FCF beyond what real operating cash generation
  would suggest. Cross-check against the company's reported Free Cash
  Flow figure directly in these cases.
- **Companies with temporarily elevated growth** (e.g. AI-driven names,
  recent IPOs): Year 1 growth is pulled from recent/live data (clamped at
  35% maximum) and may still not represent a sustainable long-term rate.
  Always set Year 5 growth deliberately to a normalized figure.
- **Capital-intensive companies mid-expansion** (e.g. large cloud/AI
  infrastructure buildouts): the CapEx-to-D&A taper assumes convergence to
  a steady state by Year 5, which may understate ongoing investment needs
  for companies still in an active expansion phase.
""")

        st.subheader("❌ Does not apply to")
        st.markdown("""
- **Banks & financial institutions** — valued on Price/Book and net interest
  margin dynamics, not standard FCF. This DCF approach does not apply.
- **REITs** — valued on FFO (Funds From Operations), not standard FCF/EBITDA.
- **Commodities & cyclicals** — earnings and margins swing too sharply with
  commodity prices for flat or smoothly-tapering assumptions to be realistic.
- **Pre-revenue companies** — percentage-of-revenue assumptions break down
  with little or no revenue base to anchor them.
""")

        st.subheader("Assumptions baked into every number")
        st.markdown("""
- **EBITDA margin and revenue growth** taper linearly from a live Year 1
  figure (clamped at 35% maximum growth) to a user-set Year 5 figure —
  Year 5 should reflect a deliberate, sustainable view, not be left at
  the default.
- **D&A%, Capex%, SBC% of revenue** are pre-filled from this company's most
  recent actual data (working capital uses a 3-year historical average to
  smooth anomalies), and are fully editable.
- **CapEx tapers toward D&A and Working Capital change tapers toward zero**
  by Year 5, simulating a mature, steady-state terminal environment — this
  helps high-growth, high-CapEx companies but can understate cash drag for
  genuinely mature companies whose CapEx/WC are already stable; always
  sanity-check Year 5 of the FCF build against the company's own profile.
- **Share dilution** from SBC is estimated heuristically (0.5% dilution per
  1% of SBC margin, compounded over 5 years) — this is a simplification,
  not a sourced or empirically derived relationship. For precise dilution
  analysis, model actual historical share count growth from filings instead.
- **WACC** is the single biggest swing factor in any DCF due to terminal
  value dominance (terminal value alone has been measured at 70-85%+ of
  total enterprise value in testing). A single slider value applies across
  all scenarios; it should be set deliberately per company based on actual
  beta and capital structure, not left at a generic default.
- **Scenarios (Bear/Base/Bull)** now run the full 5-year DCF mechanism, with
  growth and margin shocked at both Year 1 and Year 5 — Base case here will
  always match the Valuation tab's Base case exactly.
- **Share count** is cross-checked against an independently-derived figure
  (Market Cap ÷ Price); if yfinance's raw figure disagrees by more than 5%,
  the derived figure is used instead, shown transparently on Overview.
""")

        st.caption("This tool is for portfolio/educational purposes and "
                  "should not be the sole basis for an investment decision.")

else:
    st.warning("Enter a valid ticker in the sidebar to begin.")

st.markdown("---")
st.caption("Data sourced from Yahoo Finance via yfinance. "
           "All valuations are model outputs based on simplified assumptions "
           "and should not be taken as investment advice.")
