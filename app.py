import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

# Page config
st.set_page_config(page_title="PATEL DCF", page_icon="N", layout="wide", initial_sidebar_state="collapsed")


def inject_styles():
    """Set the visual language for the valuation workspace."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --ink: #17232b;
            --muted: #66747a;
            --paper: #f7f7f2;
            --panel: #ffffff;
            --line: #d9dfdc;
            --teal: #0e766e;
            --coral: #e56b54;
        }

        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1rem; }
        .block-container { max-width: 1180px; padding: 2.5rem 3rem 5rem; }
        h1, h2, h3 { color: var(--ink); letter-spacing: -0.03em; }
        h1 { font-weight: 800; font-size: clamp(2.4rem, 5vw, 4.8rem); line-height: .98; }
        h2 { font-size: 2rem; }
        h3 { font-size: 1.2rem; }
        p, label, .stCaption { color: var(--muted); }
        .brand-mark { color: var(--teal); font-family: 'DM Mono', monospace; font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 1.25rem; }
        .hero { border-bottom: 1px solid var(--line); padding: 1rem 0 2.5rem; margin-bottom: 1.25rem; }
        .hero h1 { max-width: 760px; margin: 0; }
        .hero p { max-width: 620px; font-size: 1.05rem; line-height: 1.7; margin: 1.25rem 0 0; }
        .eyebrow { color: var(--coral); font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; }
        .stat-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 1.15rem 1.25rem; min-height: 116px; }
        .stat-card .label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
        .stat-card .value { color: var(--ink); font-size: 1.65rem; font-weight: 800; margin-top: .55rem; }
        div[data-testid="stTabs"] > div:first-child { border-bottom: 1px solid var(--line); gap: .5rem; }
        button[data-baseweb="tab"] { color: var(--muted); font-weight: 700; padding: .85rem 1rem; }
        button[data-baseweb="tab"][aria-selected="true"] { color: var(--teal); }
        div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 1rem; }
        div[data-testid="stMetricLabel"] p { font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; }
        .stButton > button[kind="primary"] { background: var(--teal); border: 0; border-radius: 6px; font-weight: 800; }
        .stButton > button[kind="primary"]:hover { background: #095b55; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); }
        code { font-family: 'DM Mono', monospace; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Credit spreads lookup table (from Damodaran, updated January 2025)
CREDIT_SPREADS = [
    {"Rating": "Aaa/AAA", "Spread": 0.45},
    {"Rating": "Aa2/AA", "Spread": 0.60},
    {"Rating": "A1/A+", "Spread": 0.77},
    {"Rating": "A2/A", "Spread": 0.85},
    {"Rating": "A3/A-", "Spread": 0.95},
    {"Rating": "Baa2/BBB", "Spread": 1.20},
    {"Rating": "Ba1/BB+", "Spread": 1.55},
    {"Rating": "Ba2/BB", "Spread": 1.83},
    {"Rating": "B1/B+", "Spread": 2.61},
    {"Rating": "B2/B", "Spread": 3.00},
    {"Rating": "B3/B-", "Spread": 4.42},
    {"Rating": "Caa/CCC", "Spread": 7.28},
    {"Rating": "Ca2/CC", "Spread": 10.10},
    {"Rating": "C2/C", "Spread": 15.50},
    {"Rating": "D2/D", "Spread": 19.00},
]

# ============================================================================
# Helper Functions
# ============================================================================

def get_credit_spread(rating):
    """Returns the credit spread for a given rating."""
    for entry in CREDIT_SPREADS:
        if entry["Rating"].lower() == rating.lower():
            return entry["Spread"] / 100
    return None


def _positive_number(value):
    """Return a positive numeric value, or None for missing market data."""
    try:
        number = float(value)
        return number if np.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def _non_negative_number(value):
    """Return a non-negative numeric value, preserving legitimate zeroes."""
    try:
        number = float(value)
        return number if np.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def _latest_statement_value(statement, names):
    """Read the latest available value for one of several statement labels."""
    for name in names:
        if name not in statement.index:
            continue
        values = pd.to_numeric(statement.loc[name], errors="coerce").dropna()
        if not values.empty:
            return _non_negative_number(values.sort_index().iloc[-1])
    return None


def get_company_market_data(ticker_symbol):
    """Resolve core company data with fallbacks for hosted Yahoo responses."""
    ticker = yf.Ticker(ticker_symbol)

    try:
        ticker_info = ticker.info
    except Exception:
        ticker_info = {}

    try:
        fast_info = ticker.fast_info
    except Exception:
        fast_info = {}

    try:
        balance_sheet = ticker.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()

    shares_outstanding = next(
        (
            value for value in (
                ticker_info.get("sharesOutstanding"),
                fast_info.get("shares"),
            )
            if _positive_number(value) is not None
        ),
        None,
    )
    shares_outstanding = _positive_number(shares_outstanding)

    current_price = next(
        (
            value for value in (
                ticker_info.get("currentPrice"),
                ticker_info.get("regularMarketPrice"),
                fast_info.get("last_price"),
            )
            if _positive_number(value) is not None
        ),
        None,
    )

    if current_price is None:
        try:
            recent_prices = ticker.history(period="5d", auto_adjust=False)["Close"].dropna()
            if not recent_prices.empty:
                current_price = _positive_number(recent_prices.iloc[-1])
        except Exception:
            current_price = None

    market_cap = next(
        (
            value for value in (
                ticker_info.get("marketCap"),
                fast_info.get("market_cap"),
            )
            if _positive_number(value) is not None
        ),
        None,
    )
    market_cap = _positive_number(market_cap)
    if market_cap is None and current_price is not None and shares_outstanding is not None:
        market_cap = current_price * shares_outstanding

    total_debt = _non_negative_number(ticker_info.get("totalDebt"))
    if (total_debt is None or total_debt == 0) and not balance_sheet.empty:
        total_debt = _latest_statement_value(balance_sheet, ["Total Debt"])
        if total_debt is None:
            long_term_debt = _latest_statement_value(
                balance_sheet, ["Long Term Debt And Capital Lease Obligation"]
            )
            current_debt = _latest_statement_value(
                balance_sheet, ["Current Debt And Capital Lease Obligation", "Current Debt"]
            )
            if long_term_debt is not None or current_debt is not None:
                total_debt = (long_term_debt or 0) + (current_debt or 0)

    total_cash = _non_negative_number(ticker_info.get("totalCash"))
    if (total_cash is None or total_cash == 0) and not balance_sheet.empty:
        total_cash = _latest_statement_value(
            balance_sheet,
            ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
        )

    if shares_outstanding is None or market_cap is None:
        raise ValueError(
            f"Yahoo Finance did not return usable price and share data for {ticker_symbol}. "
            "Try again in a moment or check the ticker symbol."
        )

    return {
        "company_name": ticker_info.get("longName", ticker_symbol),
        "market_cap": market_cap,
        "shares_outstanding": shares_outstanding,
        "total_debt": total_debt if total_debt is not None else 0,
        "total_cash": total_cash if total_cash is not None else 0,
        "current_price": current_price,
    }

def calculate_beta(ticker_symbol, index_symbol="^GSPC"):
    """Calculate beta using OLS regression on 5 years of monthly returns."""
    stock_data = yf.download(ticker_symbol, period='5y', interval='1mo', progress=False)['Close']
    index_data = yf.download(index_symbol, period='5y', interval='1mo', progress=False)['Close']

    stock_returns = stock_data.pct_change().dropna()
    index_returns = index_data.pct_change().dropna()

    aligned_data = pd.concat([stock_returns, index_returns], axis=1).dropna()
    aligned_data.columns = ['Stock', 'Market']

    X = sm.add_constant(aligned_data['Market'])
    model = sm.OLS(aligned_data['Stock'], X)
    results = model.fit()

    beta = results.params['Market']
    r_squared = results.rsquared

    return beta, r_squared, results, aligned_data


def _statement_series(statement, names, default=np.nan):
    """Return the first available statement line item under common Yahoo labels."""
    for name in names:
        if name in statement.columns:
            return pd.to_numeric(statement[name], errors="coerce")
    return pd.Series(default, index=statement.index, dtype="float64")


def get_historical_data(ticker_symbol):
    """Fetch and process historical financial data for a company."""
    ticker = yf.Ticker(ticker_symbol)

    # Get financial statements
    income_statement = ticker.financials.T.sort_index()
    balance_sheet = ticker.balance_sheet.T.sort_index()
    cash_flows = ticker.cashflow.T.sort_index()

    # Normalize statement labels because Yahoo omits some lines for industries
    # such as biotech and uses alternate names for depreciation.
    income_statement['Total Revenue'] = _statement_series(
        income_statement, ['Total Revenue', 'Operating Revenue']
    )
    income_statement['EBIT'] = _statement_series(
        income_statement, ['EBIT', 'Operating Income']
    )
    gross_profit = _statement_series(income_statement, ['Gross Profit'])
    if gross_profit.isna().all():
        cost_of_revenue = _statement_series(income_statement, ['Cost Of Revenue'])
        gross_profit = income_statement['Total Revenue'] - cost_of_revenue
    income_statement['Gross Profit'] = gross_profit
    income_statement['Gross Margin'] = income_statement['Gross Profit'] / income_statement['Total Revenue']
    income_statement['EBIT Margin'] = income_statement['EBIT'] / income_statement['Total Revenue']
    income_statement['Revenue Growth'] = income_statement['Total Revenue'].pct_change()
    income_statement['EBIT Growth'] = income_statement['EBIT'].pct_change()

    # Get effective tax rate
    if 'Tax Rate For Calcs' in income_statement.columns:
        eff_tax_rate = pd.to_numeric(income_statement['Tax Rate For Calcs'], errors='coerce')
    else:
        # Calculate from tax provision and pretax income if available
        if 'Tax Provision' in income_statement.columns and 'Pretax Income' in income_statement.columns:
            eff_tax_rate = (
                pd.to_numeric(income_statement['Tax Provision'], errors='coerce')
                / pd.to_numeric(income_statement['Pretax Income'], errors='coerce')
            )
        else:
            eff_tax_rate = pd.Series([0.21] * len(income_statement), index=income_statement.index)
    eff_tax_rate = eff_tax_rate.replace([np.inf, -np.inf], np.nan).fillna(0.21)

    # Calculate NWC from balance sheet
    cash_col = 'Cash Cash Equivalents And Short Term Investments' if 'Cash Cash Equivalents And Short Term Investments' in balance_sheet.columns else 'Cash And Cash Equivalents'
    debt_col = 'Current Debt And Capital Lease Obligation' if 'Current Debt And Capital Lease Obligation' in balance_sheet.columns else 'Current Debt'

    current_assets = _statement_series(balance_sheet, ['Current Assets'], default=0)
    current_liabilities = _statement_series(balance_sheet, ['Current Liabilities'], default=0)
    cash = _statement_series(balance_sheet, [cash_col], default=0)
    current_debt = _statement_series(balance_sheet, [debt_col], default=0)
    balance_sheet["Adj CA"] = current_assets - cash
    balance_sheet["Adj CL"] = current_liabilities - current_debt
    balance_sheet["NWC"] = balance_sheet["Adj CA"] - balance_sheet["Adj CL"]
    balance_sheet["Ch in NWC"] = balance_sheet["NWC"].diff()

    # Process cash flows
    cash_flows['Capital Expenditure'] = -_statement_series(
        cash_flows, ['Capital Expenditure'], default=0
    )
    cash_flows['Depreciation And Amortization'] = _statement_series(
        cash_flows,
        ['Depreciation And Amortization', 'Depreciation Amortization Depletion', 'Depreciation'],
        default=0,
    )

    # Merge for reinvestment calculation
    merged_cf = cash_flows.join(balance_sheet[["NWC", "Ch in NWC"]])
    merged_cf['Reinvestment'] = (
        merged_cf['Capital Expenditure']
        - merged_cf['Depreciation And Amortization']
        + merged_cf['Ch in NWC']
    )

    # Calculate NOPAT
    income_statement['NOPAT'] = income_statement['EBIT'] * (1 - eff_tax_rate)

    # Build summary stats dataframe
    df_stats = income_statement[['Revenue Growth', 'EBIT Growth', 'Gross Margin', 'EBIT Margin']].copy()
    df_stats['Eff Tax Rate'] = eff_tax_rate
    df_stats['NOPAT'] = income_statement['NOPAT']
    df_stats['Reinvestment'] = merged_cf['Reinvestment']
    df_stats['Reinv Rate'] = merged_cf['Reinvestment'] / income_statement['NOPAT']

    # Sustainable growth is driven by reinvestment and the return earned on
    # the capital supporting the business.
    total_debt = _statement_series(balance_sheet, ['Total Debt'], default=0)
    if total_debt.eq(0).all():
        long_term_debt = _statement_series(
            balance_sheet, ['Long Term Debt And Capital Lease Obligation'], default=0
        )
        current_debt = _statement_series(
            balance_sheet, ['Current Debt And Capital Lease Obligation', 'Current Debt'], default=0
        )
        total_debt = long_term_debt + current_debt
    equity = _statement_series(
        balance_sheet,
        ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'],
    )
    cash = _statement_series(
        balance_sheet,
        ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents'],
        default=0,
    )
    invested_capital = total_debt + equity - cash
    invested_capital = invested_capital.reindex(income_statement.index)
    df_stats['Return on Capital'] = income_statement['NOPAT'] / invested_capital

    return {
        'income_statement': income_statement,
        'balance_sheet': balance_sheet,
        'cash_flows': cash_flows,
        'merged_cf': merged_cf,
        'df_stats': df_stats
    }


def calculate_terminal_growth_rate(historical_data, wacc):
    """Estimate sustainable terminal growth from historical reinvestment and ROC."""
    stats = historical_data['df_stats'].replace([np.inf, -np.inf], np.nan)
    average_reinvestment = stats['Reinv Rate'].dropna().mean()
    average_return_on_capital = stats['Return on Capital'].dropna().mean()

    if np.isfinite(average_reinvestment) and np.isfinite(average_return_on_capital):
        raw_growth = average_reinvestment * average_return_on_capital
        method = 'Average reinvestment rate × average return on capital'
    else:
        raw_growth = stats['Revenue Growth'].dropna().mean()
        method = 'Average historical revenue growth (fallback)'

    if not np.isfinite(raw_growth):
        raw_growth = 0.03
        method = 'Default long-term growth assumption'

    # Keep the Gordon Growth Model stable: terminal growth must remain below WACC.
    maximum_growth = min(0.05, max(0.0, wacc - 0.005))
    recommended_growth = min(max(float(raw_growth), 0.0), maximum_growth)
    return recommended_growth, average_reinvestment, average_return_on_capital, method

def get_ltm_revenue(ticker_symbol):
    """Get Last Twelve Months revenue from quarterly data."""
    ticker = yf.Ticker(ticker_symbol)
    quarterly_data = ticker.quarterly_financials.T.sort_index()
    ltm_data = quarterly_data.iloc[-4:]
    revenue_column = 'Total Revenue' if 'Total Revenue' in ltm_data.columns else 'Operating Revenue'
    ltm_revenue = ltm_data[revenue_column].sum()
    most_recent_date = ltm_data.index[-1]
    return ltm_revenue, most_recent_date

def build_dcf_projections(ltm_revenue, most_recent_date, growth_rates, ebit_margins, reinv_rates, eff_tax_rate):
    """Build projections dataframe for DCF model."""
    time_horizon = len(growth_rates)

    # Generate future dates
    freq_dict = {1:'YE-JAN', 2:'YE-FEB', 3:'YE-MAR', 4:'YE-APR', 5:'YE-MAY', 6:'YE-JUN',
                 7:'YE-JUL', 8:'YE-AUG', 9:'YE-SEP', 10:'YE-OCT', 11:'YE-NOV', 12:'YE-DEC'}
    f = freq_dict[most_recent_date.month]
    new_dates = pd.date_range(start=most_recent_date, periods=time_horizon + 1, freq=f)

    # Create projections dataframe
    projections = pd.DataFrame(index=new_dates[1:], data={
        'Revenue Growth': growth_rates,
        'EBIT Margin': ebit_margins,
        'Reinv Rate': reinv_rates
    })

    # Project revenue
    projected_revenue = [ltm_revenue]
    for i in range(time_horizon):
        new_revenue = projected_revenue[i] * (1 + growth_rates[i])
        projected_revenue.append(new_revenue)
    projected_revenue.pop(0)

    projections['Revenue'] = projected_revenue
    projections['EBIT'] = projections['Revenue'] * projections['EBIT Margin']
    projections['NOPAT'] = projections['EBIT'] * (1 - eff_tax_rate)
    projections['FCF'] = projections['NOPAT'] * (1 - projections['Reinv Rate'])
    projections['T'] = range(1, time_horizon + 1)

    return projections

def calculate_dcf_valuation(projections, wacc, terminal_growth, total_debt, total_cash, shares_outstanding):
    """Calculate DCF valuation and implied share price."""
    time_horizon = len(projections)

    # Discount FCF to present
    projections['Discounted_FCF'] = projections['FCF'] / (1 + wacc) ** projections['T']
    pv_fcf = projections['Discounted_FCF'].sum()

    # Terminal value using Gordon Growth Model
    final_fcf = projections.iloc[-1]['FCF']
    terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** time_horizon

    # Enterprise value and equity value
    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value - total_debt + total_cash
    share_price = equity_value / shares_outstanding

    return {
        'pv_fcf': pv_fcf,
        'terminal_value': terminal_value,
        'pv_terminal': pv_terminal,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'share_price': share_price,
        'projections': projections
    }

# ============================================================================
# Page: Home
# ============================================================================

def render_home():
    st.markdown(
        """
        <section class="hero">
            <div class="brand-mark">PATEL DCF / Equity Research Workspace</div>
            <div class="eyebrow">A disciplined view of value</div>
            <h1>Turn operating signals into an investable point of view.</h1>
            <p>PATEL DCF brings cost of capital, company history, and forward cash flow into one focused valuation workspace.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    intro_col, signal_col = st.columns([1.6, 1], gap="large")
    with intro_col:
        st.markdown("#### Three lenses. One valuation.")
        st.markdown(
            "Move from market assumptions to business fundamentals, then test what the company could be worth. "
            "Each stage leaves you with inputs you can carry into the next."
        )
    with signal_col:
        st.markdown("<div class='stat-card'><div class='label'>Model architecture</div><div class='value'>WACC → FCF → EV</div><p>Transparent inputs, visible assumptions, no black box.</p></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)
    step_cols = st.columns(3, gap="medium")
    steps = [
        ("01 / Capital", "WACC Calculator", "Estimate beta, cost of equity, cost of debt, and the discount rate."),
        ("02 / Evidence", "Historical Analysis", "Read growth, margins, working capital, and reinvestment across reported periods."),
        ("03 / Value", "DCF Model", "Project free cash flow, discount terminal value, and compare implied value with price."),
    ]
    for column, (eyebrow, title, description) in zip(step_cols, steps):
        with column:
            st.markdown(f"<div class='stat-card'><div class='eyebrow'>{eyebrow}</div><h3>{title}</h3><p>{description}</p></div>", unsafe_allow_html=True)

    st.markdown("### The core mechanics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**WACC**")
        st.latex(r"WACC = w_E \times k_E + w_D \times k_D \times (1-t)")

        st.markdown("**Cost of Equity (CAPM)**")
        st.latex(r"k_E = r_f + \beta \times EMRP")

    with col2:
        st.markdown("**Free Cash Flow**")
        st.latex(r"FCF = NOPAT \times (1 - ReinvestmentRate)")

        st.markdown("**Terminal Value**")
        st.latex(r"TV = \frac{FCF_{final} \times (1 + g)}{WACC - g}")

    st.markdown("<div style='height: .5rem'></div>", unsafe_allow_html=True)
    st.caption("Built for MGT6534 | Market and financial statement data from Yahoo Finance")

# ============================================================================
# Page: WACC Calculator
# ============================================================================

def render_wacc():
    st.header("WACC Calculator")
    st.markdown("Calculate the Weighted Average Cost of Capital for any publicly traded company.")

    # Inputs in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        ticker_symbol = st.text_input("Ticker Symbol", value="MSFT").upper()
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0, max_value=20.0, value=4.5, step=0.1,
            help="Enter the current 10-year Treasury yield"
        ) / 100

    with col2:
        emrp = st.number_input(
            "Equity Market Risk Premium (%)",
            min_value=0.0, max_value=20.0, value=5.0, step=0.1
        ) / 100
        rating_options = [entry["Rating"] for entry in CREDIT_SPREADS]
        firm_rating = st.selectbox("Credit Rating", options=rating_options, index=0)

    with col3:
        marg_tax_rate = st.number_input(
            "Marginal Tax Rate (%)",
            min_value=0.0, max_value=50.0, value=25.0, step=1.0
        ) / 100

    calculate_button = st.button("Calculate WACC", type="primary")

    if calculate_button:
        with st.spinner(f"Fetching data for {ticker_symbol}..."):
            try:
                company_data = get_company_market_data(ticker_symbol)
                company_name = company_data["company_name"]
                market_cap = company_data["market_cap"]
                total_debt = company_data["total_debt"]

                st.subheader(f"{company_name} ({ticker_symbol})")

                # Calculate weights
                w_E = market_cap / (market_cap + total_debt)
                w_D = total_debt / (market_cap + total_debt)

                # Calculate beta
                beta, r_squared, reg_results, return_data = calculate_beta(ticker_symbol)

                # Calculate cost of equity (CAPM)
                cost_of_equity = risk_free_rate + beta * emrp

                # Calculate cost of debt
                credit_spread = get_credit_spread(firm_rating)
                cost_of_debt = risk_free_rate + credit_spread

                # Calculate WACC
                wacc = (w_E * cost_of_equity) + (w_D * cost_of_debt * (1 - marg_tax_rate))

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Capital Structure**")
                    st.metric("Market Cap", f"${market_cap/1e9:,.2f}B")
                    st.metric("Total Debt", f"${total_debt/1e9:,.2f}B")
                    st.metric("Equity Weight (w_E)", f"{w_E:.2%}")
                    st.metric("Debt Weight (w_D)", f"{w_D:.2%}")

                with col2:
                    st.markdown("**Beta Estimation**")
                    st.metric("Beta (β)", f"{beta:.3f}")
                    st.metric("R-squared", f"{r_squared:.3f}")
                    st.caption("Based on 5-year monthly returns vs S&P 500")

                st.divider()

                col3, col4 = st.columns(2)

                with col3:
                    st.markdown("**Cost of Equity**")
                    st.latex(r"k_E = r_f + \beta \times EMRP")
                    st.write(f"k_E = {risk_free_rate:.2%} + {beta:.3f} × {emrp:.2%}")
                    st.metric("Cost of Equity (k_E)", f"{cost_of_equity:.2%}")

                with col4:
                    st.markdown("**Cost of Debt**")
                    st.latex(r"k_D = r_f + spread")
                    st.write(f"k_D = {risk_free_rate:.2%} + {credit_spread:.2%}")
                    st.metric("Cost of Debt (k_D)", f"{cost_of_debt:.2%}")
                    st.caption(f"Credit Rating: {firm_rating}")

                st.divider()

                # Final WACC
                st.markdown("**Weighted Average Cost of Capital**")
                st.latex(r"WACC = w_E \times k_E + w_D \times k_D \times (1-t)")
                st.write(f"WACC = {w_E:.2%} × {cost_of_equity:.2%} + {w_D:.2%} × {cost_of_debt:.2%} × (1 - {marg_tax_rate:.0%})")

                st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>WACC = {wacc:.2%}</h1>", unsafe_allow_html=True)

                # Estimate sustainable terminal growth from historical fundamentals.
                try:
                    historical_data = get_historical_data(ticker_symbol)
                    terminal_growth, average_reinvestment, average_return_on_capital, growth_method = calculate_terminal_growth_rate(
                        historical_data, wacc
                    )
                    st.session_state["dcf_terminal_growth_rate"] = terminal_growth * 100
                    st.session_state["dcf_wacc_rate"] = wacc * 100

                    st.markdown("**Terminal Growth Rate**")
                    st.latex(r"g = Reinvestment	ext{ Rate} \times Return	ext{ on Capital}")
                    growth_col1, growth_col2, growth_col3 = st.columns(3)
                    with growth_col1:
                        st.metric("Average Reinvestment Rate", f"{average_reinvestment:.2%}")
                    with growth_col2:
                        st.metric("Average Return on Capital", f"{average_return_on_capital:.2%}")
                    with growth_col3:
                        st.metric("Recommended Terminal Growth", f"{terminal_growth:.2%}")
                    st.caption(f"{growth_method}. The recommended rate is capped at 5% and kept below WACC for model stability.")
                    st.info("This recommended terminal growth rate is now loaded into the DCF Model tab.")
                except Exception as growth_error:
                    st.warning(f"WACC calculated, but terminal growth could not be estimated from historicals: {growth_error}")

                # Summary table
                st.markdown("**Summary**")
                summary_df = pd.DataFrame({
                    "Metric": ["Risk-Free Rate", "Beta", "EMRP", "Cost of Equity",
                              "Credit Spread", "Cost of Debt", "Equity Weight",
                              "Debt Weight", "Tax Rate", "WACC"],
                    "Value": [f"{risk_free_rate:.2%}", f"{beta:.3f}", f"{emrp:.2%}",
                             f"{cost_of_equity:.2%}", f"{credit_spread:.2%}",
                             f"{cost_of_debt:.2%}", f"{w_E:.2%}", f"{w_D:.2%}",
                             f"{marg_tax_rate:.0%}", f"{wacc:.2%}"]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# Page: Historical Analysis
# ============================================================================

def render_historical():
    st.header("Historical Analysis")
    st.markdown("Analyze historical financial performance: growth rates, margins, and reinvestment.")

    # Inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        ticker_symbol = st.text_input("Ticker Symbol", value="MSFT", key="hist_ticker").upper()

    with col2:
        scale_options = {"Millions ($M)": 1_000_000, "Billions ($B)": 1_000_000_000}
        scale_choice = st.selectbox("Display Scale", options=list(scale_options.keys()))
        scale_factor = scale_options[scale_choice]
        scale_name = "$M" if scale_factor == 1_000_000 else "$B"

    analyze_button = st.button("Analyze Financials", type="primary")

    if analyze_button:
        with st.spinner(f"Fetching financial data for {ticker_symbol}..."):
            try:
                ticker = yf.Ticker(ticker_symbol)
                company_name = ticker.info.get('longName', ticker_symbol)

                st.subheader(f"{company_name} ({ticker_symbol})")

                data = get_historical_data(ticker_symbol)
                income_statement = data['income_statement']
                balance_sheet = data['balance_sheet']
                merged_cf = data['merged_cf']
                df_stats = data['df_stats']

                # Section 1: Revenue and EBIT
                st.markdown("### Income Statement Highlights")

                is_display = income_statement[['Total Revenue', 'EBIT', 'Gross Profit']].copy()
                is_display = is_display / scale_factor
                is_display.index = is_display.index.strftime('%Y-%m-%d')
                st.dataframe(is_display.style.format("{:,.2f}"), use_container_width=True)
                st.caption(f"Values in {scale_name}")

                # Section 2: Growth Rates and Margins
                st.markdown("### Growth Rates & Margins")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Growth Rates**")
                    growth_df = df_stats[['Revenue Growth', 'EBIT Growth']].copy()
                    growth_df.index = growth_df.index.strftime('%Y-%m-%d')
                    st.dataframe(growth_df.style.format("{:.2%}"), use_container_width=True)

                with col2:
                    st.markdown("**Margins**")
                    margin_df = df_stats[['Gross Margin', 'EBIT Margin']].copy()
                    margin_df.index = margin_df.index.strftime('%Y-%m-%d')
                    st.dataframe(margin_df.style.format("{:.2%}"), use_container_width=True)

                # Section 3: NWC Analysis
                st.markdown("### Working Capital Analysis")

                nwc_df = balance_sheet[['NWC', 'Ch in NWC']].copy()
                nwc_df = nwc_df / scale_factor
                nwc_df.index = nwc_df.index.strftime('%Y-%m-%d')
                st.dataframe(nwc_df.style.format("{:,.2f}"), use_container_width=True)
                st.caption(f"Values in {scale_name}")

                # Section 4: Reinvestment
                st.markdown("### Reinvestment Analysis")

                reinv_df = merged_cf[['Capital Expenditure', 'Depreciation And Amortization', 'Ch in NWC', 'Reinvestment']].copy()
                reinv_df = reinv_df / scale_factor
                reinv_df.index = reinv_df.index.strftime('%Y-%m-%d')
                st.dataframe(reinv_df.style.format("{:,.2f}"), use_container_width=True)
                st.caption(f"Reinvestment = CapEx - D&A + Change in NWC | Values in {scale_name}")

                # Section 5: Summary Statistics
                st.markdown("### Summary: Key Valuation Metrics")

                summary_display = df_stats.copy()
                summary_display.index = summary_display.index.strftime('%Y-%m-%d')

                # Format different columns appropriately
                format_dict = {
                    'Revenue Growth': '{:.2%}',
                    'EBIT Growth': '{:.2%}',
                    'Gross Margin': '{:.2%}',
                    'EBIT Margin': '{:.2%}',
                    'Eff Tax Rate': '{:.2%}',
                    'NOPAT': '{:,.0f}',
                    'Reinvestment': '{:,.0f}',
                    'Reinv Rate': '{:.2%}'
                }

                # Scale NOPAT and Reinvestment for display
                summary_display['NOPAT'] = summary_display['NOPAT'] / scale_factor
                summary_display['Reinvestment'] = summary_display['Reinvestment'] / scale_factor

                st.dataframe(summary_display.style.format(format_dict), use_container_width=True)
                st.caption(f"NOPAT and Reinvestment in {scale_name}")

                # Averages for quick reference
                st.markdown("### Historical Averages (for projections)")

                avg_col1, avg_col2, avg_col3, avg_col4 = st.columns(4)

                with avg_col1:
                    avg_rev_growth = df_stats['Revenue Growth'].mean()
                    st.metric("Avg Revenue Growth", f"{avg_rev_growth:.2%}")

                with avg_col2:
                    avg_ebit_margin = df_stats['EBIT Margin'].mean()
                    st.metric("Avg EBIT Margin", f"{avg_ebit_margin:.2%}")

                with avg_col3:
                    avg_tax_rate = df_stats['Eff Tax Rate'].mean()
                    st.metric("Avg Eff Tax Rate", f"{avg_tax_rate:.2%}")

                with avg_col4:
                    avg_reinv_rate = df_stats['Reinv Rate'].mean()
                    st.metric("Avg Reinv Rate", f"{avg_reinv_rate:.2%}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check that the ticker symbol is valid and has sufficient financial data.")

# ============================================================================
# Page: DCF Model
# ============================================================================

def render_dcf():
    st.header("DCF Model")
    st.markdown("Build the discounted cash flow valuation model to derive an implied share price.")

    # === Section 1: Basic Inputs ===
    st.markdown("### Company & Valuation Inputs")

    if "dcf_wacc_rate" not in st.session_state:
        st.session_state["dcf_wacc_rate"] = 9.0
    if "dcf_terminal_growth_rate" not in st.session_state:
        st.session_state["dcf_terminal_growth_rate"] = 3.0

    col1, col2, col3 = st.columns(3)

    with col1:
        ticker_symbol = st.text_input("Ticker Symbol", value="MSFT", key="dcf_ticker").upper()

    with col2:
        wacc = st.number_input(
            "WACC (%)",
            min_value=1.0, max_value=30.0, step=0.25, key="dcf_wacc_rate",
            help="From WACC Calculator tab"
        ) / 100

    with col3:
        terminal_growth = st.number_input(
            "Terminal Growth Rate (%)",
            min_value=0.0, max_value=5.0, step=0.25, key="dcf_terminal_growth_rate",
            help="Calculated in the WACC Calculator from historical reinvestment and return on capital"
        ) / 100

    col4, col5 = st.columns(2)

    with col4:
        eff_tax_rate = st.number_input(
            "Effective Tax Rate (%)",
            min_value=0.0, max_value=50.0, value=21.0, step=1.0,
            help="From Historical Analysis tab"
        ) / 100

    with col5:
        scale_options = {"Millions ($M)": 1_000_000, "Billions ($B)": 1_000_000_000}
        scale_choice = st.selectbox("Display Scale", options=list(scale_options.keys()), key="dcf_scale")
        scale_factor = scale_options[scale_choice]
        scale_name = "$M" if scale_factor == 1_000_000 else "$B"

    # === Section 2: Projection Assumptions ===
    st.markdown("### Projection Assumptions")
    st.caption("Enter comma-separated values for each year of your projection period (e.g., 10 years)")

    # Default values matching the notebook
    default_growth = "20, 15, 15, 15, 10, 10, 10, 8, 8, 6"
    default_margin = "46, 46, 46, 46, 46, 46, 46, 46, 46, 46"
    default_reinv = "40, 40, 30, 20, 20, 20, 20, 20, 20, 20"

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        growth_input = st.text_input(
            "Revenue Growth Rates (%)",
            value=default_growth,
            help="Annual revenue growth rates for each projection year"
        )

    with col_b:
        margin_input = st.text_input(
            "EBIT Margins (%)",
            value=default_margin,
            help="EBIT margin for each projection year"
        )

    with col_c:
        reinv_input = st.text_input(
            "Reinvestment Rates (%)",
            value=default_reinv,
            help="Reinvestment rate for each projection year"
        )

    # Parse inputs
    try:
        growth_rates = [float(x.strip()) / 100 for x in growth_input.split(',')]
        ebit_margins = [float(x.strip()) / 100 for x in margin_input.split(',')]
        reinv_rates = [float(x.strip()) / 100 for x in reinv_input.split(',')]

        # Validate lengths match
        if not (len(growth_rates) == len(ebit_margins) == len(reinv_rates)):
            st.error("All projection inputs must have the same number of values.")
            return

        time_horizon = len(growth_rates)
        st.caption(f"Projection period: {time_horizon} years")

    except ValueError:
        st.error("Please enter valid comma-separated numbers for all projection inputs.")
        return

    # === Run Valuation Button ===
    run_dcf = st.button("Run DCF Valuation", type="primary")

    if run_dcf:
        with st.spinner(f"Running DCF valuation for {ticker_symbol}..."):
            try:
                # Get company data
                company_data = get_company_market_data(ticker_symbol)
                company_name = company_data["company_name"]
                shares_outstanding = company_data["shares_outstanding"]
                total_debt = company_data["total_debt"]
                total_cash = company_data["total_cash"]
                current_price = company_data["current_price"]

                # Get LTM Revenue
                ltm_revenue, most_recent_date = get_ltm_revenue(ticker_symbol)

                st.subheader(f"{company_name} ({ticker_symbol})")

                # Display starting data
                st.markdown("### Starting Point: LTM Data")
                col_ltm1, col_ltm2, col_ltm3 = st.columns(3)
                with col_ltm1:
                    st.metric("LTM Revenue", f"{ltm_revenue/scale_factor:,.2f} {scale_name}")
                with col_ltm2:
                    st.metric("Most Recent Quarter", most_recent_date.strftime('%Y-%m-%d'))
                with col_ltm3:
                    st.metric("Shares Outstanding", f"{shares_outstanding/1e6:,.2f}M")

                # Build projections
                projections = build_dcf_projections(
                    ltm_revenue, most_recent_date,
                    growth_rates, ebit_margins, reinv_rates, eff_tax_rate
                )

                # Calculate valuation
                valuation = calculate_dcf_valuation(
                    projections, wacc, terminal_growth,
                    total_debt, total_cash, shares_outstanding
                )

                # === Display Projections ===
                st.markdown("### Projected Financials")

                # Assumptions table
                st.markdown("**Assumptions by Year**")
                assumptions_df = projections[['Revenue Growth', 'EBIT Margin', 'Reinv Rate']].copy()
                assumptions_df.index = assumptions_df.index.strftime('%Y')
                st.dataframe(
                    assumptions_df.style.format("{:.1%}"),
                    use_container_width=True
                )

                # Dollar projections table
                st.markdown("**Projected Values**")
                dollar_cols = ['Revenue', 'EBIT', 'NOPAT', 'FCF', 'Discounted_FCF']
                dollar_df = valuation['projections'][dollar_cols].copy() / scale_factor
                dollar_df.index = dollar_df.index.strftime('%Y')
                st.dataframe(
                    dollar_df.style.format("{:,.2f}"),
                    use_container_width=True
                )
                st.caption(f"Values in {scale_name}")

                # === Terminal Value ===
                st.markdown("### Terminal Value")

                st.latex(r"TV = \frac{FCF_{final} \times (1 + g)}{WACC - g}")

                final_fcf = projections.iloc[-1]['FCF']
                st.write(f"TV = ({final_fcf/scale_factor:,.2f} × (1 + {terminal_growth:.2%})) / ({wacc:.2%} - {terminal_growth:.2%})")

                col_tv1, col_tv2 = st.columns(2)
                with col_tv1:
                    st.metric("Terminal Value", f"{valuation['terminal_value']/scale_factor:,.2f} {scale_name}")
                with col_tv2:
                    st.metric("PV of Terminal Value", f"{valuation['pv_terminal']/scale_factor:,.2f} {scale_name}")

                # === Enterprise Value ===
                st.markdown("### Enterprise Value")

                st.latex(r"EV = \sum_{t=1}^{n} \frac{FCF_t}{(1+WACC)^t} + \frac{TV}{(1+WACC)^n}")

                col_ev1, col_ev2, col_ev3 = st.columns(3)
                with col_ev1:
                    st.metric("PV of FCFs", f"{valuation['pv_fcf']/scale_factor:,.2f} {scale_name}")
                with col_ev2:
                    st.metric("PV of Terminal Value", f"{valuation['pv_terminal']/scale_factor:,.2f} {scale_name}")
                with col_ev3:
                    st.metric("Enterprise Value", f"{valuation['enterprise_value']/scale_factor:,.2f} {scale_name}")

                # === Equity Value & Share Price ===
                st.markdown("### Equity Value & Implied Share Price")

                st.latex(r"Equity\ Value = EV - Debt + Cash")
                st.latex(r"Share\ Price = \frac{Equity\ Value}{Shares\ Outstanding}")

                col_eq1, col_eq2, col_eq3 = st.columns(3)
                with col_eq1:
                    st.metric("Total Debt", f"{total_debt/scale_factor:,.2f} {scale_name}")
                with col_eq2:
                    st.metric("Total Cash", f"{total_cash/scale_factor:,.2f} {scale_name}")
                with col_eq3:
                    st.metric("Equity Value", f"{valuation['equity_value']/scale_factor:,.2f} {scale_name}")

                st.divider()

                # Final share price display
                implied_price = valuation['share_price']

                col_price1, col_price2, col_price3 = st.columns(3)
                with col_price1:
                    st.metric("Current Market Price", f"${current_price:,.2f}")
                with col_price2:
                    st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>${implied_price:,.2f}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center;'><strong>Implied Share Price</strong></p>", unsafe_allow_html=True)
                with col_price3:
                    upside = (implied_price - current_price) / current_price * 100
                    color = "green" if upside > 0 else "red"
                    st.metric("Upside/Downside", f"{upside:+.1f}%")

                # === Sensitivity Analysis ===
                st.markdown("### Sensitivity Analysis")
                st.caption("See how implied share price changes with different WACC assumptions")

                wacc_low = st.number_input("Low WACC (%)", value=(wacc*100 - 1.5), step=0.25, key="wacc_low") / 100
                wacc_high = st.number_input("High WACC (%)", value=(wacc*100 + 1.5), step=0.25, key="wacc_high") / 100

                sensitivity_results = []
                for w in [wacc_low, wacc, wacc_high]:
                    proj_copy = build_dcf_projections(
                        ltm_revenue, most_recent_date,
                        growth_rates, ebit_margins, reinv_rates, eff_tax_rate
                    )
                    val = calculate_dcf_valuation(
                        proj_copy, w, terminal_growth,
                        total_debt, total_cash, shares_outstanding
                    )
                    sensitivity_results.append({
                        'WACC': f"{w:.2%}",
                        'Enterprise Value': f"{val['enterprise_value']/scale_factor:,.2f}",
                        'Equity Value': f"{val['equity_value']/scale_factor:,.2f}",
                        'Share Price': f"${val['share_price']:,.2f}",
                        'vs Current': f"{((val['share_price'] - current_price) / current_price * 100):+.1f}%"
                    })

                sensitivity_df = pd.DataFrame(sensitivity_results)
                st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)

                # === Summary Table ===
                st.markdown("### Valuation Summary")
                summary_data = {
                    "Metric": [
                        "LTM Revenue",
                        "Projection Period",
                        "WACC",
                        "Terminal Growth Rate",
                        "Effective Tax Rate",
                        "PV of FCFs",
                        "Terminal Value",
                        "PV of Terminal Value",
                        "Enterprise Value",
                        "Less: Total Debt",
                        "Plus: Total Cash",
                        "Equity Value",
                        "Shares Outstanding",
                        "Implied Share Price",
                        "Current Market Price",
                        "Upside/Downside"
                    ],
                    "Value": [
                        f"{ltm_revenue/scale_factor:,.2f} {scale_name}",
                        f"{time_horizon} years",
                        f"{wacc:.2%}",
                        f"{terminal_growth:.2%}",
                        f"{eff_tax_rate:.2%}",
                        f"{valuation['pv_fcf']/scale_factor:,.2f} {scale_name}",
                        f"{valuation['terminal_value']/scale_factor:,.2f} {scale_name}",
                        f"{valuation['pv_terminal']/scale_factor:,.2f} {scale_name}",
                        f"{valuation['enterprise_value']/scale_factor:,.2f} {scale_name}",
                        f"({total_debt/scale_factor:,.2f}) {scale_name}",
                        f"{total_cash/scale_factor:,.2f} {scale_name}",
                        f"{valuation['equity_value']/scale_factor:,.2f} {scale_name}",
                        f"{shares_outstanding/1e6:,.2f}M",
                        f"${implied_price:,.2f}",
                        f"${current_price:,.2f}",
                        f"{upside:+.1f}%"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check that the ticker symbol is valid and has sufficient financial data.")

# ============================================================================
# Main App with Tabs
# ============================================================================

# Shared visual shell
inject_styles()
st.markdown("<div class='brand-mark'>PATEL DCF</div>", unsafe_allow_html=True)

# Create tabs
tab_home, tab_wacc, tab_historical, tab_dcf = st.tabs([
    "🏠 Home",
    "1️⃣ WACC Calculator",
    "2️⃣ Historical Analysis",
    "3️⃣ DCF Model"
])

with tab_home:
    render_home()

with tab_wacc:
    render_wacc()

with tab_historical:
    render_historical()

with tab_dcf:
    render_dcf()
