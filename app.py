"""
Equity Research & Valuation Platform
v6 — FINAL, REALISTIC VERSION

Honest scope: a simplified DCF tool. No generalized DCF tool perfectly
matches every company - SBC treatment, working capital direction, and
WACC all require company-specific judgment that a generic formula can't
fully automate. This version reflects everything validated through testing:

  - FCF = EBIT*(1-tax) + D&A + SBC - Capex - WC_change  (tax bug fixed,
    SBC addback added - this was the single biggest gap for SaaS names)
  - Revenue growth: Year1 (from yfinance) -> Year5 (user), tapered
  - EBITDA margin: Year1 (from yfinance) -> Year5 (user), tapered
  - Capex%, D&A%, SBC%, WC% are flat sliders, pre-filled from the
    company's actual last reported year, fully user-editable
  - WC% can be NEGATIVE (a cash source - common for subscription
    businesses with deferred revenue, e.g. Snowflake)
  - WACC default 6.8%, Terminal Growth default 2.0%, both sliders
  - Default ticker: MSFT (stable, well-covered, sane first impression)
  - Scenarios (Tab 4): simple single-year snapshot, NOT a full 5-year
    re-discount - growth x0.85/1.15, margin x0.90/1.05, same WACC
  - Limitations tab documents what this tool does NOT handle well

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
    FCF = EBIT*(1-tax) + D&A + SBC - Capex - WC_change
    da_pct, sbc_pct, capex_pct, wc_pct are flat % of revenue, fixed for
    all 5 years. wc_pct may be NEGATIVE (cash source).
    growth_path and margin_path are 5-element lists (already tapered).
    """
    fcfs = []
    revenue = revenue_base
    for g, margin in zip(growth_path, margin_path):
        revenue  = revenue * (1 + g)
        ebitda   = revenue * margin
        da       = revenue * da_pct
        ebit     = ebitda - da
        nopat    = ebit * (1 - tax_rate)
        sbc      = revenue * sbc_pct
        capex    = revenue * capex_pct
        delta_wc = revenue * wc_pct
        fcf      = nopat + da + sbc - capex - delta_wc
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


def sensitivity_table(fcfs, wacc_range, growth_range, net_debt=0, shares=1):
    """Same mechanism as a standard Excel sensitivity table: the FCF
    stream is fixed, only WACC and terminal growth vary across the grid."""
    wacc_grid, g_grid = np.meshgrid(wacc_range, growth_range)
    results = np.zeros_like(wacc_grid)
    for i in range(wacc_grid.shape[0]):
        for j in range(wacc_grid.shape[1]):
            w = wacc_grid[i, j]
            g = g_grid[i, j]
            if w > g:
                r = dcf_value(fcfs, w, g, net_debt)
                results[i, j] = r['equity_value'] / shares if shares else 0
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


def reverse_dcf(fcfs, wacc, net_debt, shares, target_price):
    g_range   = np.linspace(0.01, wacc - 0.005, 200)
    best_g    = None
    best_diff = float('inf')
    for g in g_range:
        result        = dcf_value(fcfs, wacc, g, net_debt)
        implied_price = result['equity_value'] / shares
        diff          = abs(implied_price - target_price)
        if diff < best_diff:
            best_diff = diff
            best_g    = g
    return best_g


def estimate_wacc(info):
    """CAPM-based estimate, DISPLAY ONLY. The WACC slider is the real input."""
    beta = info.get('beta') or 1.0
    rf, erp, kd, tax = 0.04, 0.05, 0.04, 0.21
    ke = rf + beta * erp
    debt   = info.get('totalDebt', 0) or 0
    equity = info.get('marketCap', 0) or 0
    total  = debt + equity
    if total == 0:
        return ke
    we, wd = equity / total, debt / total
    return we * ke + wd * kd * (1 - tax)


def get_actuals(info, income, cashflow):
    """
    Pull this company's most recent actual EBITDA margin, D&A%, Capex%,
    SBC%, and WC% from real data, as STARTING POINTS for the sliders.
    Falls back to reasonable generic defaults if data is missing.
    These are DISPLAYED and EDITABLE - not hidden, not forced.
    """
    revenue = info.get('totalRevenue', 0) or 0

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
            if 'Change In Working Capital' in cashflow.index:
                v = cashflow.loc['Change In Working Capital', latest_col]
                if pd.notna(v) and revenue:
                    # Sign matters: yfinance reports this as the cash
                    # flow statement does - a positive value here means
                    # it was a SOURCE of cash, so we keep the sign as-is
                    # (negative wc_pct = source, matches our formula)
                    wc_pct = -(v) / revenue
            if 'Stock Based Compensation' in cashflow.index:
                v = cashflow.loc['Stock Based Compensation', latest_col]
                if pd.notna(v) and revenue:
                    sbc_pct = abs(v) / revenue
    except Exception:
        pass

    da_pct        = max(min(da_pct, 0.20), 0.0)
    capex_pct     = max(min(capex_pct, 0.30), 0.0)
    wc_pct        = max(min(wc_pct, 0.15), -0.20)
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
ticker = st.sidebar.text_input("Enter Ticker", value="PEP").upper()

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
actual_growth = float(round(max(min(raw_growth, 80.0), -20.0), 1))

est_wacc = estimate_wacc(info) if data_loaded else 0.068

st.sidebar.header("Valuation Assumptions")
wacc       = st.sidebar.slider("WACC (%)", 3.0, 15.0, 6.8, 0.1) / 100
terminal_g = st.sidebar.slider("Terminal Growth (%)", 0.0, 5.0, 2.0, 0.25) / 100
tax_rate   = st.sidebar.slider("Tax Rate (%)", 0.0, 40.0, 21.0, 1.0) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Growth & Margin (Year 1 → Year 5)")
st.sidebar.caption("Year 1 pulled live. Year 5 is your view of the "
                  "sustainable, normalized level. Tapers between them.")

yr1_growth = st.sidebar.slider("Year 1 Revenue Growth (%)", -20.0, 80.0, actual_growth, 1.0) / 100
yr5_growth = st.sidebar.slider("Year 5 Revenue Growth (%)", 1.0, 30.0, 8.0, 1.0) / 100

raw_margin    = actuals['ebitda_margin'] * 100
yr1_margin = st.sidebar.slider(
    "Year 1 EBITDA Margin (%)", -30.0, 60.0,
    float(round(max(min(raw_margin, 60.0), -30.0), 1)), 1.0
) / 100
yr5_margin = st.sidebar.slider(
    "Year 5 EBITDA Margin (%)", 0.0, 60.0,
    float(round(min(max(raw_margin, 5.0) + 5.0, 60.0), 1)), 1.0
) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("FCF Build — Flat % of Revenue")
st.sidebar.caption("Pre-filled from this company's last actual year. "
                  "Edit if the live-pulled figure looks unusual for "
                  "this specific company.")

da_pct    = st.sidebar.slider("D&A (% of Revenue)", 0.0, 20.0,
                              float(round(actuals['da_pct']*100, 1)), 0.5) / 100
capex_pct = st.sidebar.slider("Capex (% of Revenue)", 0.0, 30.0,
                              float(round(actuals['capex_pct']*100, 1)), 0.5) / 100
sbc_pct   = st.sidebar.slider("Stock-Based Comp (% of Revenue)", 0.0, 40.0,
                              float(round(actuals['sbc_pct']*100, 1)), 0.5) / 100
wc_pct    = st.sidebar.slider("Working Capital Change (% of Revenue)", -20.0, 15.0,
                              float(round(actuals['wc_pct']*100, 1)), 0.5) / 100
st.sidebar.caption("Negative Working Capital % = a cash SOURCE (e.g. "
                  "subscription businesses billing upfront via deferred "
                  "revenue, like Snowflake). Positive = a cash drag.")

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

    revenue_base  = info.get('totalRevenue', 0) / 1e6
    net_debt      = (info.get('totalDebt', 0) - info.get('totalCash', 0)) / 1e6
    shares        = info.get('sharesOutstanding', 1) / 1e6
    current_price = info.get('currentPrice', 0) or 0

    kpis = calculate_kpis(info, income)

    fcfs = fcf_projection(
        revenue_base, GROWTH_PATH, MARGIN_PATH,
        da_pct, sbc_pct, capex_pct, wc_pct, tax_rate
    )

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
            st.metric("Market Cap", f"${info.get('marketCap',0)/1e9:,.1f}B")
        with col3:
            st.metric("Revenue", f"${revenue_base:,.0f}M")
        with col4:
            gm = kpis.get('gross_margin')
            st.metric("Gross Margin", f"{gm:.1%}" if gm else "N/A")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            pe = kpis.get('pe_ratio')
            st.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
        with col6:
            rg = kpis.get('revenue_growth')
            st.metric("Revenue Growth", f"{rg:.1%}" if rg else "N/A")
        with col7:
            roe = kpis.get('roe')
            st.metric("ROE", f"{roe:.1%}" if roe else "N/A")
        with col8:
            st.metric("Est. WACC", f"{est_wacc:.1%}",
                      help="CAPM-based estimate, reference only. The WACC "
                           "slider is what the valuation actually uses.")

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
                      f"— all flat, all editable in the sidebar.")

        result    = dcf_value(fcfs, wacc, terminal_g, net_debt)
        intrinsic = result['equity_value'] / shares if shares else 0
        delta     = ((intrinsic / current_price) - 1) if current_price else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Enterprise Value", f"${result['enterprise_value']:,.0f}M")
        with col2:
            st.metric("Equity Value", f"${result['equity_value']:,.0f}M")
        with col3:
            st.metric("Intrinsic Value/Share", f"${intrinsic:,.2f}",
                      delta=f"{delta:.1%} vs market")

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

        sens = sensitivity_table(fcfs, wacc_range, growth_range, net_debt, shares)
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
        if current_price and shares:
            implied_g = reverse_dcf(fcfs, wacc, net_debt, shares, current_price)
            st.metric("Market-Implied Terminal Growth", f"{implied_g:.1%}")
            st.caption(f"At ${current_price:.2f}, the market implies terminal "
                      f"growth of **{implied_g:.1%}** given your current "
                      f"assumptions at {wacc:.1%} WACC.")
            if implied_g > 0.05:
                st.error("⚠️ Very aggressive implied growth — check assumptions.")
            elif implied_g > 0.04:
                st.warning("⚠️ Above-average growth expectations priced in.")
            else:
                st.success("✅ Conservative, justifiable growth assumption.")

    # ----------------------------------------------------------------
    # TAB 4 — SCENARIOS (simple, single-year snapshot)
    # ----------------------------------------------------------------
    with tab4:
        st.header("Scenario Analysis")
        st.caption("A quick single-year snapshot — NOT a full 5-year "
                  "re-discount. Bear: this year's growth ×0.85, margin "
                  "×0.90. Bull: growth ×1.15, margin ×1.05. Same WACC.")

        ebitda_y1 = revenue_base * (1 + yr1_growth) * yr1_margin

        def quick_price(growth_mult, margin_mult):
            rev1   = revenue_base * (1 + yr1_growth * growth_mult)
            ebitda = rev1 * (yr1_margin * margin_mult)
            da     = rev1 * da_pct
            ebit   = ebitda - da
            nopat  = ebit * (1 - tax_rate)
            sbc    = rev1 * sbc_pct
            capex  = rev1 * capex_pct
            dwc    = rev1 * wc_pct
            fcf1   = nopat + da + sbc - capex - dwc
            # simple one-year-forward perpetuity using same WACC/terminal_g
            value = fcf1 * (1 + terminal_g) / (wacc - terminal_g)
            equity = value - net_debt
            return equity / shares if shares else 0

        bear_price = quick_price(0.85, 0.90)
        base_price = quick_price(1.00, 1.00)
        bull_price = quick_price(1.15, 1.05)

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
                st.subheader("P/S Ratio Comparison")
                st.caption("💡 **Why P/S?** Many growth companies have low or "
                          "negative earnings, making P/E unreliable. P/S "
                          "compares price to revenue — harder to manipulate, "
                          "almost always positive.")
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
        st.info("ℹ️ Calibrated for mature profitable companies. Pre-profit "
                "growth companies score lower on profitability/capital "
                "efficiency even if the business is high quality.")

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

        st.subheader("What this tool does NOT handle well")
        st.markdown("""
- **Banks & financial institutions** — valued on Price/Book and net interest
  margin, not standard FCF. This DCF approach does not apply.
- **REITs** — valued on FFO (Funds From Operations), not standard FCF/EBITDA.
- **Commodities & cyclicals** — earnings swing too sharply with commodity
  prices for flat-growth, flat-cost assumptions to be realistic.
- **Pre-revenue companies** — percentage-of-revenue assumptions break down
  with little or no revenue base.
""")

        st.subheader("Assumptions baked into every number")
        st.markdown(f"""
- **Capex%, D&A%, SBC%, Working Capital%** are pre-filled from this
  company's most recently reported actual year, then held FLAT for the
  entire 5-year forecast. They are editable — if a live-pulled figure
  looks unusual for this specific company (this happened with Capex% for
  some large-capex companies during testing), check it and adjust.
- **Revenue growth and EBITDA margin** taper linearly from a live Year 1
  figure to a user-set Year 5 figure. Year 1 reflects current/recent
  performance; Year 5 should reflect your view of the sustainable,
  normalized level — these are NOT always the same number, especially
  for high-growth or AI-driven names with temporarily elevated growth.
- **Working Capital%** can be set negative for subscription businesses
  with deferred revenue (a cash source, not a drag) — this matters a lot
  for SaaS companies and was a real gap found during testing.
- **Stock-Based Compensation** is added back as non-cash — material for
  high-growth tech/SaaS, usually small for mature companies.
- **WACC** is the single biggest swing factor in any DCF. The Est. WACC
  on Overview is a rough CAPM reference — it can be meaningfully off for
  very low-risk, very large companies. Verify independently when possible.
- **Scenarios (Bear/Base/Bull)** are a simplified single-year snapshot,
  not independently researched 5-year forecasts.
- **No simplified DCF tool perfectly matches every company** — names with
  unusual economics (heavy SBC, deferred revenue, temporarily elevated
  growth) require more judgment than flat assumptions can capture. Treat
  outputs as a starting point for analysis, not a precise answer.
""")

        st.caption("This tool is for portfolio/educational purposes and "
                  "should not be the sole basis for an investment decision.")

else:
    st.warning("Enter a valid ticker in the sidebar to begin.")

st.markdown("---")
st.caption("Data sourced from Yahoo Finance via yfinance. "
           "All valuations are model outputs based on simplified assumptions "
           "and should not be taken as investment advice.")

st.markdown(
    "<div style='text-align:center; padding:8px; background-color:#1a2b4a; "
    "color:white; border-radius:6px; font-size:13px; margin-top:10px;'>"
    "🟢 Running app.py — v6 FINAL (tax fix, SBC addback, negative WC, "
    "Year1\u2192Year5 taper, WACC 6.8% / Terminal 2.0% defaults, MSFT default)"
    "</div>",
    unsafe_allow_html=True
)