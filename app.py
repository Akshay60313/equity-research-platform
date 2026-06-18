"""
Equity Research & Valuation Platform
Complete app with all fixes

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
            # Add inside your existing st.markdown CSS block
        .stApp p { color: #1a2b4a; }
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    [data-testid="stSidebar"] { background-color: #1a2b4a; }
    [data-testid="stSidebar"] * { color: white !important; }
    h1, h2, h3 { color: #1a2b4a; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Equity Research & Valuation Platform")
st.caption("Live financial analysis, DCF valuation, and scenario modelling for any public company")


# ============================================================
# CORE FINANCE FUNCTIONS
# ============================================================

def fcf_projection(revenue_base, growth_rates, margins, da_pct, capex_pct, wc_pct):
    fcfs = []
    revenue = revenue_base
    for g, margin in zip(growth_rates, margins):
        revenue  = revenue * (1 + g)
        ebitda   = revenue * margin
        capex    = revenue * capex_pct
        delta_wc = revenue * wc_pct
        fcfs.append(round(ebitda - capex - delta_wc, 1))
    return fcfs


def dcf_value(fcfs, wacc, g, net_debt=0):
    if wacc <= g:
        raise ValueError("WACC must exceed terminal growth rate")
    pv_fcfs          = sum([fcf / (1 + wacc)**i for i, fcf in enumerate(fcfs, 1)])
    terminal_value   = (fcfs[-1] * (1 + g)) / (wacc - g)
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
    df.index.name   = 'Growth'
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
}


def get_peers_for(ticker):
    return PEER_MAP.get(ticker, [])


# ============================================================
# SIDEBAR — TICKER FIRST, DATA PULL, THEN SLIDERS
# ============================================================

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Enter Ticker", value="SNOW").upper()

# Pull data immediately after ticker — BEFORE sliders
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

# Read actual margin from live data to pre-populate slider
raw_margin    = (info.get('ebitdaMargins') or 0.08) * 100
actual_margin = float(round(max(min(raw_margin, 40.0), -5.0), 1))

st.sidebar.header("Valuation Assumptions")
wacc          = st.sidebar.slider("WACC (%)", 5.0, 15.0, 10.0, 0.5)
growth        = st.sidebar.slider("Terminal Growth (%)", 1.0, 5.0, 3.0, 0.25)
exit_multiple = st.sidebar.slider("Exit EV/EBITDA Multiple", 5.0, 60.0, 20.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Assumptions")

yr1_growth = st.sidebar.slider("Year 1 Growth (%)", 1.0, 60.0, 20.0, 1.0) / 100
yr5_growth = st.sidebar.slider("Year 5 Growth (%)", 1.0, 30.0, 10.0, 1.0) / 100

yr1_margin = st.sidebar.slider(
    "Year 1 EBITDA Margin (%)", -5.0, 40.0,
    actual_margin, 1.0
) / 100

yr5_margin = st.sidebar.slider(
    "Year 5 EBITDA Margin (%)", 0.0, 50.0,
    float(round(min(actual_margin + 2.0, 50.0), 1)), 1.0
) / 100

DA_PCT    = 0.05
CAPEX_PCT = 0.04
WC_PCT    = 0.02

DEFAULT_GROWTH_RATES = [
    yr1_growth,
    yr1_growth * 0.87,
    yr1_growth * 0.75,
    yr1_growth * 0.63,
    yr5_growth
]
DEFAULT_MARGINS = [
    yr1_margin,
    yr1_margin + (yr5_margin - yr1_margin) * 0.25,
    yr1_margin + (yr5_margin - yr1_margin) * 0.50,
    yr1_margin + (yr5_margin - yr1_margin) * 0.75,
    yr5_margin
]


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Financials", "Valuation", "Scenarios", "Peers", "Quality Score"]
)

if data_loaded:

    revenue_base  = info.get('totalRevenue', 0) / 1e6
    net_debt      = (info.get('totalDebt', 0) - info.get('totalCash', 0)) / 1e6
    shares        = info.get('sharesOutstanding', 1) / 1e6
    current_price = info.get('currentPrice', 0) or 0

    assumptions = {
        'growth_rates': DEFAULT_GROWTH_RATES,
        'margins':      DEFAULT_MARGINS,
        'da_pct':       DA_PCT,
        'capex_pct':    CAPEX_PCT,
        'wc_pct':       WC_PCT,
        'wacc':         wacc / 100,
        'g':            growth / 100
    }

    fcfs = fcf_projection(
        revenue_base,
        assumptions['growth_rates'], assumptions['margins'],
        assumptions['da_pct'], assumptions['capex_pct'], assumptions['wc_pct']
    )

    kpis = calculate_kpis(info, income)

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
            de = kpis.get('debt_to_equity')
            st.metric("Debt/Equity", f"{de:.0f}" if de else "N/A")

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

        result    = dcf_value(fcfs, assumptions['wacc'], assumptions['g'], net_debt)
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

        st.subheader("Sensitivity Analysis (Intrinsic Value per Share)")
        wacc_range   = np.linspace(max(assumptions['wacc'] - 0.02, 0.01),
                                   assumptions['wacc'] + 0.02, 5)
        growth_range = np.linspace(max(assumptions['g'] - 0.01, 0.005),
                                   assumptions['g'] + 0.01, 3)

        sens = sensitivity_table(fcfs, wacc_range, growth_range, net_debt, shares)
        st.dataframe(sens.style.format("${:,.2f}"), use_container_width=True)

        fig_heat = go.Figure(go.Heatmap(
            z=sens.values,
            x=sens.columns.tolist(),
            y=sens.index.tolist(),
            colorscale='RdYlGn',
            text=sens.values,
            texttemplate='$%{text:.2f}',
            hovertemplate='WACC: %{x}<br>Growth: %{y}<br>Price: $%{z:.2f}<extra></extra>'
        ))
        fig_heat.update_layout(
            title='Intrinsic Value per Share — WACC vs Terminal Growth',
            xaxis_title='WACC', yaxis_title='Terminal Growth', height=400
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Reverse DCF — Implied Growth")
        if current_price and shares:
            implied_g = reverse_dcf(fcfs, assumptions['wacc'],
                                    net_debt, shares, current_price)
            st.metric("Market-Implied Terminal Growth", f"{implied_g:.1%}")
            st.caption(f"""
**What this means:** At ${current_price:.2f}, the market implies terminal growth of 
**{implied_g:.1%}** using your current margin and growth assumptions at {wacc}% WACC.
If this seems high, check your margin assumptions match the company's actual EBITDA margin.
Long-run GDP growth is typically 2-3%.
""")
            if implied_g > 0.05:
                st.error("⚠️ Very aggressive implied growth — check your margin assumptions "
                         "match the company's actual EBITDA margin.")
            elif implied_g > 0.04:
                st.warning("⚠️ Above-average growth expectations priced in.")
            else:
                st.success("✅ Conservative, justifiable growth assumption.")

    # ----------------------------------------------------------------
    # TAB 4 — SCENARIOS
    # ----------------------------------------------------------------
    with tab4:
        st.header("Scenario Analysis")

        scenarios = {
            'Bear': {
                'growth_rates': [g * 0.55 for g in DEFAULT_GROWTH_RATES],
                'margins':      [max(m * 0.60, 0.03) for m in DEFAULT_MARGINS],
                'wacc': assumptions['wacc'] + 0.02,
                'g': 0.015
            },
            'Base': {
                'growth_rates': DEFAULT_GROWTH_RATES,
                'margins':      DEFAULT_MARGINS,
                'wacc': assumptions['wacc'],
                'g':    assumptions['g']
            },
            'Bull': {
                'growth_rates': [g * 1.30 for g in DEFAULT_GROWTH_RATES],
                'margins':      [m * 1.30 for m in DEFAULT_MARGINS],
                'wacc': max(assumptions['wacc'] - 0.01, 0.01),
                'g': 0.035
            }
        }

        def run_scenario(s):
            f = fcf_projection(
                revenue_base, s['growth_rates'], s['margins'],
                assumptions['da_pct'], assumptions['capex_pct'], assumptions['wc_pct']
            )
            r = dcf_value(f, s['wacc'], s['g'], net_debt)
            return r['equity_value'] / shares if shares else 0

        scenario_results = {name: run_scenario(s) for name, s in scenarios.items()}

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bear Case", f"${scenario_results['Bear']:.2f}")
        with col2:
            st.metric("Base Case", f"${scenario_results['Base']:.2f}")
        with col3:
            st.metric("Bull Case", f"${scenario_results['Bull']:.2f}")

        st.write(f"Current market price: **${current_price:.2f}**")

        fig_range = go.Figure()
        fig_range.add_trace(go.Bar(
            x=['Bear', 'Base', 'Bull'],
            y=[max(scenario_results['Bear'], 0),
               scenario_results['Base'],
               scenario_results['Bull']],
            marker_color=['#E24B4A', '#378ADD', '#1D9E75']
        ))
        if current_price:
            fig_range.add_hline(y=current_price, line_dash='dash',
                                annotation_text='Current Price',
                                line_color='black')
        fig_range.update_layout(title='Valuation Range by Scenario',
                                yaxis_title='Price per Share ($)')
        st.plotly_chart(fig_range, use_container_width=True)

        st.subheader("5-Year Return Model")

        revenue_y5 = revenue_base
        for g in DEFAULT_GROWTH_RATES:
            revenue_y5 = revenue_y5 * (1 + g)
        ebitda_margin_y5 = DEFAULT_MARGINS[-1]

        use_rev = st.checkbox(
            "Use EV/Revenue multiple (recommended for pre-profit growth companies)",
            value=False
        )

        if use_rev:
            rev_mult = st.slider("Exit EV/Revenue Multiple", 3.0, 50.0, 15.0, 0.5)
            exit_ev  = revenue_y5 * rev_mult
        else:
            exit_ev = (revenue_y5 * ebitda_margin_y5) * exit_multiple

        exit_equity_val = exit_ev - net_debt
        exit_price_val  = exit_equity_val / shares if shares else 0
        total_ret = (exit_price_val - current_price) / current_price if current_price else 0
        cagr_val  = (exit_price_val / current_price) ** (1/5) - 1 if current_price else 0

        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.metric("Projected Exit Price", f"${exit_price_val:,.2f}")
        with rcol2:
            st.metric("Total Return (5yr)", f"{total_ret:.1%}")
        with rcol3:
            st.metric("CAGR", f"{cagr_val:.1%}")

    # ----------------------------------------------------------------
    # TAB 5 — PEERS
    # ----------------------------------------------------------------
    with tab5:
        st.header("Peer Comparison")
        st.info("ℹ️ Peer comparison available for: "
                "**SNOW** (Snowflake), **AAPL** (Apple), **PEP** (PepsiCo).")

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
                fig_peers = px.bar(
                    comp.reset_index(), x='Ticker', y='P/S',
                    title='P/S Ratio Comparison', color='Ticker'
                )
                st.plotly_chart(fig_peers, use_container_width=True)
        else:
            st.info(f"No peer mapping defined for {ticker} yet.")

    # ----------------------------------------------------------------
    # TAB 6 — QUALITY SCORE
    # ----------------------------------------------------------------
    with tab6:
        st.header("Business Quality Score")
        st.info("""
ℹ️ **Scoring note:** Calibrated for mature profitable companies.
Pre-profit growth companies will score lower on profitability and capital efficiency
even if the business is high quality. Focus on **growth** and **leverage** scores
as primary signals for such companies.
""")
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
            x=list(scores.keys()),
            y=list(scores.values()),
            marker_color=score_colours
        ))
        fig_score.update_layout(
            title="Score Breakdown by Category",
            yaxis_title="Score",
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_score, use_container_width=True)

else:
    st.warning("Enter a valid ticker in the sidebar to begin.")

st.markdown("---")
st.caption("Data sourced from Yahoo Finance via yfinance. "
           "All valuations are model outputs based on simplified assumptions "
           "and should not be taken as investment advice.")