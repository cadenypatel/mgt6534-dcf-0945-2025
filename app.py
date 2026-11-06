import difflib
import io
import os

import streamlit as st
import pandas as pd
import numpy as np
import requests
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

# Fallback copies of Damodaran's published data, used only when his site
# can't be reached; the live versions are fetched below and cached for a day.
# Credit spreads: ratings.html, data as of January 2026. IcMin/IcMax are the
# interest-coverage bands that map EBIT / Interest Expense to a synthetic rating.
CREDIT_SPREADS = [
    {"Rating": "Aaa/AAA", "Spread": 0.40, "IcMin": 8.5, "IcMax": float("inf")},
    {"Rating": "Aa2/AA", "Spread": 0.55, "IcMin": 6.5, "IcMax": 8.5},
    {"Rating": "A1/A+", "Spread": 0.70, "IcMin": 5.5, "IcMax": 6.5},
    {"Rating": "A2/A", "Spread": 0.78, "IcMin": 4.25, "IcMax": 5.5},
    {"Rating": "A3/A-", "Spread": 0.89, "IcMin": 3.0, "IcMax": 4.25},
    {"Rating": "Baa2/BBB", "Spread": 1.11, "IcMin": 2.5, "IcMax": 3.0},
    {"Rating": "Ba1/BB+", "Spread": 1.38, "IcMin": 2.25, "IcMax": 2.5},
    {"Rating": "Ba2/BB", "Spread": 1.84, "IcMin": 2.0, "IcMax": 2.25},
    {"Rating": "B1/B+", "Spread": 2.75, "IcMin": 1.75, "IcMax": 2.0},
    {"Rating": "B2/B", "Spread": 3.21, "IcMin": 1.5, "IcMax": 1.75},
    {"Rating": "B3/B-", "Spread": 5.09, "IcMin": 1.25, "IcMax": 1.5},
    {"Rating": "Caa/CCC", "Spread": 8.85, "IcMin": 0.8, "IcMax": 1.25},
    {"Rating": "Ca2/CC", "Spread": 12.61, "IcMin": 0.65, "IcMax": 0.8},
    {"Rating": "C2/C", "Spread": 16.00, "IcMin": 0.2, "IcMax": 0.65},
    {"Rating": "D2/D", "Spread": 19.00, "IcMin": float("-inf"), "IcMax": 0.2},
]

# Implied US equity risk premium (trailing 12-month cash yield), October 1, 2026.
DEFAULT_EMRP = 0.0370
DEFAULT_EMRP_DATE = "Oct 2026 (stored copy)"

# US marginal corporate rate (21% federal + ~4% state, Damodaran's US default),
# used for the WACC debt tax shield.
MARGINAL_TAX_RATE = 0.25

DAMODARAN_DATA = "https://pages.stern.nyu.edu/~adamodar"
DAMODARAN_HEADERS = {"User-Agent": "Mozilla/5.0 (PATEL DCF research app)"}

# ============================================================================
# Helper Functions
# ============================================================================

def _fetch_damodaran(path):
    response = requests.get(f"{DAMODARAN_DATA}/{path}", headers=DAMODARAN_HEADERS, timeout=20)
    response.raise_for_status()
    return response.content


@st.cache_data(ttl=86400, show_spinner=False)
def get_damodaran_erp():
    """Latest monthly implied US equity risk premium from Damodaran's
    ERPbymonth.xlsx, as (premium, "Mon YYYY"). Falls back to the stored copy."""
    try:
        sheet = pd.read_excel(io.BytesIO(_fetch_damodaran("pc/implprem/ERPbymonth.xlsx")),
                              sheet_name="Historical ERP")
        sheet["ERP (T12m)"] = pd.to_numeric(sheet["ERP (T12m)"], errors="coerce")
        sheet["Start of month"] = pd.to_datetime(sheet["Start of month"], errors="coerce")
        latest = sheet.dropna(subset=["ERP (T12m)", "Start of month"]).iloc[-1]
        premium = float(latest["ERP (T12m)"])
        if 0 < premium < 0.15:
            return premium, f"{latest['Start of month']:%b %Y}"
    except Exception:
        pass
    return DEFAULT_EMRP, DEFAULT_EMRP_DATE


@st.cache_data(ttl=86400, show_spinner=False)
def get_credit_spread_table():
    """Damodaran's latest coverage-band / synthetic-rating / spread table for
    large non-financial firms. Falls back to the stored copy."""
    try:
        table = pd.read_html(io.StringIO(_fetch_damodaran(
            "New_Home_Page/datafile/ratings.html").decode("latin-1")))[0]
        rows = []
        for _, row in table.iterrows():
            low = pd.to_numeric(row.iloc[0], errors="coerce")
            high = pd.to_numeric(row.iloc[1], errors="coerce")
            spread = pd.to_numeric(str(row.iloc[3]).rstrip("%"), errors="coerce")
            if pd.isna(low) or pd.isna(high) or pd.isna(spread):
                continue
            rows.append({
                "Rating": str(row.iloc[2]).strip(),
                "Spread": float(spread),
                "IcMin": float("-inf") if low <= -1000 else float(low),
                "IcMax": float("inf") if high >= 1000 else float(high),
            })
        rows.sort(key=lambda entry: entry["IcMin"], reverse=True)
        known = {entry["Rating"] for entry in CREDIT_SPREADS}
        if len(rows) == len(CREDIT_SPREADS) and {entry["Rating"] for entry in rows} == known:
            return rows
    except Exception:
        pass
    return CREDIT_SPREADS


def get_credit_spread(rating):
    """Returns the credit spread for a given rating."""
    for entry in get_credit_spread_table():
        if entry["Rating"].lower() == rating.lower():
            return entry["Spread"] / 100
    return None


def estimate_credit_rating(interest_coverage_ratio):
    """Map an interest-coverage ratio to a synthetic rating using Damodaran's bands."""
    if interest_coverage_ratio is None or not np.isfinite(interest_coverage_ratio):
        return None
    for entry in get_credit_spread_table():
        if entry["IcMin"] <= interest_coverage_ratio < entry["IcMax"]:
            rating = entry["Rating"]
            # True Aaa/AAA is exceedingly rare in practice (a handful of issuers
            # worldwide); interest coverage alone overstates it, so cap the
            # auto-filled estimate at Aa2/AA and leave Aaa/AAA as a manual override.
            return "Aa2/AA" if rating == "Aaa/AAA" else rating
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_damodaran_industry_betas():
    """Damodaran's latest US industry unlevered betas (corrected for cash),
    keyed by industry name. Empty if his site can't be reached."""
    try:
        table = pd.read_html(io.StringIO(_fetch_damodaran(
            "New_Home_Page/datafile/Betas.html").decode("latin-1")))[0]
        table.columns = table.iloc[0]
        table = table.iloc[1:]
        names = table["Industry Name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        betas = pd.to_numeric(table["Unlevered beta corrected for cash"], errors="coerce")
        return {name: float(beta) for name, beta in zip(names, betas)
                if np.isfinite(beta) and not name.startswith("Total")}
    except Exception:
        return {}


# Yahoo Finance industry -> Damodaran industry. Unlisted industries fall back
# to the closest name match.
YAHOO_TO_DAMODARAN_INDUSTRY = {
    "Semiconductors": "Semiconductor",
    "Semiconductor Equipment & Materials": "Semiconductor Equip",
    "Software - Infrastructure": "Software (System & Application)",
    "Software - Application": "Software (System & Application)",
    "Internet Content & Information": "Software (Internet)",
    "Electronic Gaming & Multimedia": "Software (Entertainment)",
    "Information Technology Services": "Computer Services",
    "Consumer Electronics": "Computers/Peripherals",
    "Computer Hardware": "Computers/Peripherals",
    "Communication Equipment": "Telecom. Equipment",
    "Electronic Components": "Electronics (General)",
    "Scientific & Technical Instruments": "Electronics (General)",
    "Telecom Services": "Telecom. Services",
    "Internet Retail": "Retail (General)",
    "Discount Stores": "Retail (General)",
    "Department Stores": "Retail (General)",
    "Specialty Retail": "Retail (Special Lines)",
    "Apparel Retail": "Retail (Special Lines)",
    "Home Improvement Retail": "Retail (Building Supply)",
    "Grocery Stores": "Retail (Grocery and Food)",
    "Auto & Truck Dealerships": "Retail (Automotive)",
    "Restaurants": "Restaurant/Dining",
    "Beverages - Non-Alcoholic": "Beverage (Soft)",
    "Beverages - Brewers": "Beverage (Alcoholic)",
    "Beverages - Wineries & Distilleries": "Beverage (Alcoholic)",
    "Packaged Foods": "Food Processing",
    "Confectioners": "Food Processing",
    "Food Distribution": "Food Wholesalers",
    "Household & Personal Products": "Household Products",
    "Tobacco": "Tobacco",
    "Drug Manufacturers - General": "Drugs (Pharmaceutical)",
    "Drug Manufacturers - Specialty & Generic": "Drugs (Pharmaceutical)",
    "Biotechnology": "Drugs (Biotechnology)",
    "Medical Devices": "Healthcare Products",
    "Medical Instruments & Supplies": "Healthcare Products",
    "Diagnostics & Research": "Healthcare Products",
    "Healthcare Plans": "Healthcare Support Services",
    "Medical Care Facilities": "Hospitals/Healthcare Facilities",
    "Health Information Services": "Heathcare Information and Technology",
    "Oil & Gas Integrated": "Oil/Gas (Integrated)",
    "Oil & Gas E&P": "Oil/Gas (Production and Exploration)",
    "Oil & Gas Midstream": "Oil/Gas Distribution",
    "Oil & Gas Equipment & Services": "Oilfield Svcs/Equip.",
    "Oil & Gas Refining & Marketing": "Oil/Gas (Integrated)",
    "Utilities - Regulated Electric": "Utility (General)",
    "Utilities - Diversified": "Utility (General)",
    "Utilities - Regulated Water": "Utility (Water)",
    "Utilities - Renewable": "Green & Renewable Energy",
    "Aerospace & Defense": "Aerospace/Defense",
    "Airlines": "Air Transport",
    "Railroads": "Transportation (Railroads)",
    "Trucking": "Trucking",
    "Integrated Freight & Logistics": "Transportation",
    "Specialty Industrial Machinery": "Machinery",
    "Farm & Heavy Construction Machinery": "Machinery",
    "Electrical Equipment & Parts": "Electrical Equipment",
    "Engineering & Construction": "Engineering/Construction",
    "Building Products & Equipment": "Building Materials",
    "Residential Construction": "Homebuilding",
    "Auto Manufacturers": "Auto & Truck",
    "Auto Parts": "Auto Parts",
    "Specialty Chemicals": "Chemical (Specialty)",
    "Chemicals": "Chemical (Basic)",
    "Steel": "Steel",
    "Credit Services": "Financial Svcs. (Non-bank & Insurance)",
    "Asset Management": "Investments & Asset Management",
    "Capital Markets": "Brokerage & Investment Banking",
    "Banks - Diversified": "Bank (Money Center)",
    "Banks - Regional": "Banks (Regional)",
    "Insurance - Life": "Insurance (Life)",
    "Insurance - Property & Casualty": "Insurance (Prop/Cas.)",
    "Entertainment": "Entertainment",
    "Advertising Agencies": "Advertising",
    "Lodging": "Hotel/Gaming",
    "Resorts & Casinos": "Hotel/Gaming",
    "Travel Services": "Hotel/Gaming",
    "Footwear & Accessories": "Shoe",
    "Apparel Manufacturing": "Apparel",
    "Waste Management": "Environmental & Waste Services",
    "Specialty Business Services": "Business & Consumer Services",
    "Consulting Services": "Business & Consumer Services",
    "Staffing & Employment Services": "Business & Consumer Services",
    "Education & Training Services": "Education",
    "REIT - Specialty": "R.E.I.T.",
    "REIT - Industrial": "R.E.I.T.",
    "REIT - Residential": "R.E.I.T.",
    "REIT - Retail": "Retail (REITs)",
}


def _match_damodaran_industry(yahoo_industry, damodaran_names):
    """Resolve a Yahoo industry to one of Damodaran's industry names. His
    names can be truncated (e.g. "Heathcare Information and Techno"), so a
    mapped name matches on its prefix."""
    if not yahoo_industry or not damodaran_names:
        return None
    target = YAHOO_TO_DAMODARAN_INDUSTRY.get(yahoo_industry)
    if target:
        for name in damodaran_names:
            if target.startswith(name) or name.startswith(target):
                return name
    close = difflib.get_close_matches(yahoo_industry, damodaran_names, n=1, cutoff=0.6)
    return close[0] if close else None


def get_bottom_up_beta(ticker_symbol, debt_to_equity, tax_rate):
    """Damodaran bottom-up beta: industry unlevered beta relevered at the
    firm's market D/E, beta_L = beta_u x (1 + (1 - t) x D/E). Returns
    (levered beta, industry name, unlevered beta), or None if unavailable."""
    betas = get_damodaran_industry_betas()
    try:
        yahoo_industry = yf.Ticker(ticker_symbol).info.get("industry")
    except Exception:
        yahoo_industry = None
    industry = _match_damodaran_industry(yahoo_industry, list(betas))
    if industry is None:
        return None
    unlevered = betas[industry]
    return unlevered * (1 + (1 - tax_rate) * debt_to_equity), industry, unlevered


def get_interest_coverage_ratio(ticker_symbol):
    """Return the latest annual EBIT / Interest Expense ratio, or None if unavailable."""
    try:
        income_stmt = yf.Ticker(ticker_symbol).financials
        if income_stmt is None or income_stmt.empty:
            return None
        latest = income_stmt.sort_index(axis=1, ascending=False).iloc[:, 0]

        ebit = None
        # Prefer Operating Income: Yahoo's "EBIT" adds back non-operating gains.
        for label in ("Operating Income", "EBIT"):
            if label in latest.index and pd.notna(latest[label]):
                ebit = float(latest[label])
                break

        interest_expense = None
        for label in ("Interest Expense", "Interest Expense Non Operating"):
            if label in latest.index and pd.notna(latest[label]):
                interest_expense = abs(float(latest[label]))
                break

        if ebit is None or not interest_expense:
            return None
        return ebit / interest_expense
    except Exception:
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


def _reindex_nearest(series, target_index, tolerance_days=10):
    """Align a series onto target_index by nearest date within a tolerance.

    Different data sources (e.g., SEC EDGAR vs. Yahoo) can report the same
    fiscal period with slightly different period-end dates; a plain reindex
    or combine_first would miss the match entirely and leave the gap unfilled.
    Applies the same way regardless of ticker.
    """
    if series.empty:
        return pd.Series(np.nan, index=target_index, dtype="float64")
    return series.sort_index().reindex(
        target_index, method="nearest", tolerance=pd.Timedelta(days=tolerance_days)
    )


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

    # Dilution from options/RSUs/convertibles: the latest quarter's gap between
    # diluted and basic average shares, added to the current basic count.
    dilutive_shares = 0.0
    try:
        quarterly_income = ticker.quarterly_income_stmt
        if quarterly_income is not None and not quarterly_income.empty:
            latest_quarter = quarterly_income.T.sort_index()
            diluted = _statement_series(latest_quarter, ["Diluted Average Shares"]).dropna()
            basic = _statement_series(latest_quarter, ["Basic Average Shares"]).dropna()
            common_dates = diluted.index.intersection(basic.index)
            if not common_dates.empty:
                latest_date = common_dates.max()
                dilutive_shares = max(float(diluted[latest_date] - basic[latest_date]), 0.0)
    except Exception:
        dilutive_shares = 0.0

    # Long-term investments (marketable securities, equity-method stakes) are
    # non-operating assets whose income isn't in operating income.
    non_operating_assets = 0.0
    minority_interest = 0.0
    if not balance_sheet.empty:
        non_operating_assets = _latest_statement_value(balance_sheet, ["Investments And Advances"]) or 0.0
        minority_interest = _latest_statement_value(balance_sheet, ["Minority Interest"]) or 0.0

    # Foreign filers/ADRs (e.g., TSM, NVO) report statements in their home
    # currency while the share price is quoted in another; statement amounts
    # must be converted before they're compared with price or market cap.
    price_currency = (ticker_info.get("currency") or "USD").upper()
    financial_currency = (ticker_info.get("financialCurrency") or price_currency).upper()
    fx_rate = 1.0
    if financial_currency != price_currency:
        try:
            fx_history = yf.Ticker(f"{financial_currency}{price_currency}=X").history(
                period="5d", auto_adjust=False
            )["Close"].dropna()
            fx_rate = _positive_number(fx_history.iloc[-1]) if not fx_history.empty else None
        except Exception:
            fx_rate = None
        if fx_rate is None:
            raise ValueError(
                f"{ticker_symbol} reports in {financial_currency} but trades in {price_currency}, "
                "and no exchange rate was available to convert between them."
            )

    return {
        "company_name": ticker_info.get("longName", ticker_symbol),
        "price_currency": price_currency,
        "financial_currency": financial_currency,
        "fx_rate": fx_rate,
        "market_cap": market_cap,
        "shares_outstanding": shares_outstanding,
        "basic_shares_outstanding": shares_outstanding,
        "diluted_shares_outstanding": shares_outstanding + dilutive_shares,
        "total_debt": total_debt if total_debt is not None else 0,
        "total_cash": total_cash if total_cash is not None else 0,
        "non_operating_assets": non_operating_assets,
        "minority_interest": minority_interest,
        "current_price": current_price,
    }


def get_wacc_input_defaults(ticker_symbol):
    """Fetch current market assumptions and ticker-specific tax-rate/rating estimates."""
    risk_free_rate = 0.045
    try:
        treasury_history = yf.Ticker("^TNX").history(period="5d", auto_adjust=False)["Close"].dropna()
        if not treasury_history.empty:
            treasury_yield = _positive_number(treasury_history.iloc[-1])
            if treasury_yield is not None:
                risk_free_rate = treasury_yield / 100
    except Exception:
        pass

    # EMRP is a market-wide assumption: Damodaran's latest monthly implied ERP,
    # not a company-specific input; remains editable in the UI.
    equity_market_risk_premium, _ = get_damodaran_erp()

    interest_coverage_ratio = get_interest_coverage_ratio(ticker_symbol)
    credit_rating_default = estimate_credit_rating(interest_coverage_ratio) or "Baa2/BBB"

    return (
        risk_free_rate,
        equity_market_risk_premium,
        MARGINAL_TAX_RATE,
        credit_rating_default,
        interest_coverage_ratio,
    )


def get_effective_tax_rate(ticker_symbol):
    """Return one shared ticker tax-rate estimate for WACC and DCF calculations."""
    tax_rate = None
    try:
        ticker_info = yf.Ticker(ticker_symbol).info
        tax_rate = _non_negative_number(ticker_info.get("taxRateForCalcs"))
    except Exception:
        pass

    if tax_rate is None or tax_rate > 0.5:
        try:
            historical_data = get_historical_data(ticker_symbol)
            tax_rates = historical_data["df_stats"]["Eff Tax Rate"].replace([np.inf, -np.inf], np.nan).dropna()
            usable_tax_rates = tax_rates[(tax_rates >= 0) & (tax_rates <= 0.5)]
            if not usable_tax_rates.empty:
                tax_rate = float(usable_tax_rates.iloc[-1])
        except Exception:
            pass

    return tax_rate if tax_rate is not None and tax_rate <= 0.5 else 0.21


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


def adjust_beta_for_wacc(raw_beta):
    """Apply the standard Blume adjustment to a regression beta."""
    return (2 / 3) * float(raw_beta) + (1 / 3)


BETA_METHODS = {
    "Blend: company + industry (recommended)": "blend",
    "Company regression (Blume-adjusted)": "regression",
    "Industry bottom-up (Damodaran)": "bottom_up",
}


def select_beta(method, regression_beta, bottom_up):
    """Pick the CAPM beta for a BETA_METHODS choice, as (beta, explanation).
    Falls back to the regression beta when no industry beta is available."""
    if bottom_up is None:
        return regression_beta, "Industry beta unavailable; using the company regression beta."
    industry_beta = bottom_up[0]
    if method == "regression":
        return regression_beta, "Company regression beta (Blume-adjusted)."
    if method == "bottom_up":
        return industry_beta, f"Damodaran bottom-up beta for {bottom_up[1]}."
    return (regression_beta + industry_beta) / 2, (
        f"Average of company regression ({regression_beta:.3f}) and {bottom_up[1]} bottom-up ({industry_beta:.3f})."
    )


def _statement_series(statement, names, default=np.nan):
    """Return the first available statement line item under common Yahoo labels."""
    for name in names:
        if name in statement.columns:
            values = pd.to_numeric(statement[name], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(default, index=statement.index, dtype="float64")


SEC_HEADERS = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT",
        "PATEL DCF research app (contact: caden.y.patel@gmail.com)",
    )
}


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_sec_json(url):
    """Fetch and cache one SEC JSON response."""
    response = requests.get(url, headers=SEC_HEADERS, timeout=20)
    response.raise_for_status()
    return response.json()


def _get_sec_cik(ticker_symbol):
    """Resolve a ticker to its SEC zero-padded CIK."""
    ticker_data = _fetch_sec_json("https://www.sec.gov/files/company_tickers.json")
    ticker = ticker_symbol.strip().upper()
    for company in ticker_data.values():
        if company.get("ticker", "").upper() == ticker:
            return str(company["cik_str"]).zfill(10)
    raise ValueError(f"SEC did not return a CIK for ticker {ticker_symbol}.")


def _sec_fact_series(facts, tags, take_max=False):
    """Return the latest annual SEC fact for each fiscal year-end, merging tags (first tag wins, or the largest if take_max)."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    combined = pd.Series(dtype="float64")

    for tag in tags:
        fact = us_gaap.get(tag)
        if not fact:
            continue

        entries = []
        for unit_entries in fact.get("units", {}).values():
            for entry in unit_entries:
                if entry.get("form") not in ("10-K", "10-K/A") or entry.get("fp") != "FY":
                    continue
                if not entry.get("end") or "val" not in entry:
                    continue
                if entry.get("start"):
                    start = pd.to_datetime(entry["start"], errors="coerce")
                    end = pd.to_datetime(entry["end"], errors="coerce")
                    if pd.isna(start) or pd.isna(end) or (end - start).days < 300:
                        continue
                entries.append(entry)

        if not entries:
            continue

        fact_frame = pd.DataFrame(entries)
        fact_frame["filed"] = pd.to_datetime(fact_frame["filed"], errors="coerce")
        fact_frame["end"] = pd.to_datetime(fact_frame["end"], errors="coerce")
        fact_frame = fact_frame.dropna(subset=["end", "filed"])
        fact_frame = fact_frame.sort_values("filed").drop_duplicates("end", keep="last")
        current = pd.Series(
            pd.to_numeric(fact_frame["val"], errors="coerce").to_numpy(),
            index=fact_frame["end"].dt.normalize(),
            dtype="float64",
        ).sort_index()
        combined = combined.combine(current, max) if take_max else combined.combine_first(current)

    return combined.sort_index()


def get_sec_historical_statements(ticker_symbol):
    """Build annual income, balance-sheet, and cash-flow statements from SEC XBRL."""
    cik = _get_sec_cik(ticker_symbol)
    facts = _fetch_sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")

    revenue = _sec_fact_series(
        facts,
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "RegulatedAndUnregulatedOperatingRevenue",
            "SalesRevenueNet",
            "Revenues",
        ],
    )
    operating_income = _sec_fact_series(facts, ["OperatingIncomeLoss"])
    gross_profit = _sec_fact_series(facts, ["GrossProfit"])
    tax_provision = _sec_fact_series(facts, ["IncomeTaxExpenseBenefit"])
    pretax_income = _sec_fact_series(
        facts,
        [
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLoss",
        ],
    )

    current_assets = _sec_fact_series(facts, ["AssetsCurrent"])
    current_liabilities = _sec_fact_series(facts, ["LiabilitiesCurrent"])
    cash = _sec_fact_series(
        facts,
        [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
        ],
    )
    cost_of_revenue = _sec_fact_series(
        facts,
        ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"],
    )
    current_debt = _sec_fact_series(
        facts,
        ["ShortTermBorrowings", "LongTermDebtCurrent", "LongTermDebtAndFinanceLeaseObligationsCurrent"],
    )
    long_term_debt = _sec_fact_series(
        facts,
        ["LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"],
    )
    equity = _sec_fact_series(
        facts,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    )
    capital_expenditure = _sec_fact_series(
        facts,
        [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsToAcquireOtherPropertyPlantAndEquipment",
            "PaymentsForCapitalImprovements",
            "PaymentsToAcquireRealEstate",
            "PaymentsToAcquireOilAndGasPropertyAndEquipment",
            "PaymentsToDevelopRealEstateAssets",
            "PaymentsToAcquireRealEstateHeldForInvestment",
        ],
        take_max=True,
    )
    depreciation = _sec_fact_series(
        facts,
        [
            "DepreciationDepletionAndAmortization",
            "DepreciationDepletionAndAmortizationPropertyPlantEquipment",
            "DepreciationAndAmortization",
            "Depreciation",
        ],
    )

    income_index = revenue.index.union(operating_income.index).sort_values()
    balance_index = current_assets.index.union(current_liabilities.index).sort_values()
    cash_flow_index = capital_expenditure.index.union(depreciation.index).sort_values()

    income_statement = pd.DataFrame(index=income_index)
    income_statement["Total Revenue"] = revenue.reindex(income_index)
    income_statement["EBIT"] = operating_income.reindex(income_index)
    income_statement["Gross Profit"] = gross_profit.reindex(income_index)
    income_statement["Cost Of Revenue"] = cost_of_revenue.reindex(income_index)
    income_statement["Tax Provision"] = tax_provision.reindex(income_index)
    income_statement["Pretax Income"] = pretax_income.reindex(income_index)

    balance_sheet = pd.DataFrame(index=balance_index)
    # Forward/back-fill sporadic single-year gaps (e.g., a company switching
    # XBRL tags between filings) so one missing year doesn't null out the
    # entire downstream NWC/reinvestment calculation. Applies to every ticker.
    balance_sheet["Current Assets"] = current_assets.reindex(balance_index).ffill().bfill()
    balance_sheet["Current Liabilities"] = current_liabilities.reindex(balance_index).ffill().bfill()
    balance_sheet["Cash And Cash Equivalents"] = cash.reindex(balance_index).ffill().bfill().fillna(0)
    balance_sheet["Current Debt"] = current_debt.reindex(balance_index).fillna(0)
    balance_sheet["Long Term Debt And Capital Lease Obligation"] = long_term_debt.reindex(balance_index).fillna(0)
    balance_sheet["Total Debt"] = (
        balance_sheet["Current Debt"] + balance_sheet["Long Term Debt And Capital Lease Obligation"]
    )
    balance_sheet["Stockholders Equity"] = equity.reindex(balance_index).ffill().bfill()

    cash_flows = pd.DataFrame(index=cash_flow_index)
    cash_flows["Capital Expenditure"] = -capital_expenditure.abs().reindex(cash_flow_index)
    cash_flows["Depreciation And Amortization"] = depreciation.abs().reindex(cash_flow_index)

    if income_statement["Total Revenue"].dropna().shape[0] < 2:
        raise ValueError(f"SEC returned insufficient annual revenue history for {ticker_symbol}.")
    return income_statement, balance_sheet, cash_flows


def _widest_capex(statement):
    """Largest capex measure per period (positive) across Yahoo's capex line variants."""
    candidates = [
        _statement_series(statement, [name]).abs()
        for name in ('Capital Expenditure', 'Capital Expenditure Reported', 'Purchase Of PPE')
    ]
    return pd.concat(candidates, axis=1).max(axis=1, skipna=True)


def _get_yahoo_historical_statements(ticker_symbol):
    """Fetch Yahoo annual statements for companies with incomplete SEC coverage."""
    ticker = yf.Ticker(ticker_symbol)
    return ticker.financials.T.sort_index(), ticker.balance_sheet.T.sort_index(), ticker.cashflow.T.sort_index()


def _augment_with_latest_quarterly(ticker_symbol, income_statement, balance_sheet, cash_flows):
    """Augment annual statements with the latest available trailing 12 months."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        qi = None
        for getter in (lambda: ticker.quarterly_income_stmt, lambda: ticker.quarterly_financials):
            try:
                stmt = getter()
                if stmt is not None and not stmt.empty:
                    qi = stmt.T.sort_index()
                    break
            except Exception:
                continue

        if qi is None or qi.empty:
            return income_statement, balance_sheet, cash_flows

        qb = None
        try:
            if ticker.quarterly_balance_sheet is not None and not ticker.quarterly_balance_sheet.empty:
                qb = ticker.quarterly_balance_sheet.T.sort_index()
        except Exception:
            pass

        qc = None
        try:
            if ticker.quarterly_cashflow is not None and not ticker.quarterly_cashflow.empty:
                qc = ticker.quarterly_cashflow.T.sort_index()
        except Exception:
            pass

        rev_quarterly = _statement_series(qi, ['Total Revenue', 'Operating Revenue']).dropna()
        if rev_quarterly.empty:
            return income_statement, balance_sheet, cash_flows

        latest_q = rev_quarterly.index.max()
        annual_max = income_statement.index.max() if not income_statement.empty else None

        if len(rev_quarterly.loc[:latest_q]) >= 4 and (annual_max is None or latest_q > annual_max):
            tail_qi = qi.loc[:latest_q].tail(4)
            if _statement_series(tail_qi, ['Total Revenue', 'Operating Revenue']).notna().sum() < 4:
                return income_statement, balance_sheet, cash_flows
            tail_qc = qc.loc[:latest_q].tail(4) if qc is not None and not qc.empty else pd.DataFrame()
            latest_qb = qb.loc[:latest_q].iloc[-1] if qb is not None and not qb.empty else pd.Series()
            df_q = pd.DataFrame([latest_qb]) if not latest_qb.empty else pd.DataFrame()

            # Income statement flow items (sum trailing 4 quarters)
            income_statement.loc[latest_q, 'Total Revenue'] = _statement_series(tail_qi, ['Total Revenue', 'Operating Revenue']).sum(min_count=4)
            # Yahoo's quarterly "EBIT" includes non-operating income (investment
            # gains, equity-method income); operating income is the FCFF driver.
            ltm_operating_income = _statement_series(tail_qi, ['Operating Income', 'EBIT']).sum(min_count=4)
            income_statement.loc[latest_q, 'EBIT'] = ltm_operating_income
            if 'Operating Income' in income_statement.columns:
                income_statement.loc[latest_q, 'Operating Income'] = ltm_operating_income
            income_statement.loc[latest_q, 'Gross Profit'] = _statement_series(tail_qi, ['Gross Profit']).sum(min_count=4)
            income_statement.loc[latest_q, 'Cost Of Revenue'] = _statement_series(tail_qi, ['Cost Of Revenue']).sum(min_count=4)
            income_statement.loc[latest_q, 'Tax Provision'] = _statement_series(tail_qi, ['Tax Provision']).sum(min_count=4)
            income_statement.loc[latest_q, 'Pretax Income'] = _statement_series(tail_qi, ['Pretax Income']).sum(min_count=4)
            income_statement.loc[latest_q, 'Net Income'] = _statement_series(tail_qi, ['Net Income', 'Net Income Common Stockholders']).sum(min_count=4)
            income_statement.loc[latest_q, 'Interest Expense'] = _statement_series(tail_qi, ['Interest Expense', 'Interest Expense Non Operating']).sum(min_count=4)

            # Balance sheet point-in-time items (as of latest quarter)
            if not df_q.empty:
                for col in balance_sheet.columns:
                    if col == 'Current Assets':
                        balance_sheet.loc[latest_q, col] = _statement_series(df_q, ['Current Assets'], default=0).iloc[0]
                    elif col == 'Current Liabilities':
                        balance_sheet.loc[latest_q, col] = _statement_series(df_q, ['Current Liabilities'], default=0).iloc[0]
                    elif col == 'Cash And Cash Equivalents':
                        balance_sheet.loc[latest_q, col] = _statement_series(df_q, ['Cash And Cash Equivalents'], default=0).iloc[0]
                    elif col == 'Cash Cash Equivalents And Short Term Investments':
                        balance_sheet.loc[latest_q, col] = _statement_series(df_q, ['Cash Cash Equivalents And Short Term Investments'], default=0).iloc[0]
                    elif col in ('Current Debt', 'Current Debt And Capital Lease Obligation'):
                        balance_sheet.loc[latest_q, col] = _statement_series(df_q, ['Current Debt And Capital Lease Obligation', 'Current Debt'], default=0).iloc[0]
                    elif col in ('Long Term Debt And Capital Lease Obligation', 'Long Term Debt'):
                        balance_sheet.loc[latest_q, col] = _statement_series(df_q, ['Long Term Debt And Capital Lease Obligation', 'Long Term Debt'], default=0).iloc[0]
                    elif col == 'Total Debt':
                        balance_sheet.loc[latest_q, col] = _statement_series(df_q, ['Total Debt'], default=0).iloc[0]
                    elif col in ('Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'):
                        balance_sheet.loc[latest_q, col] = _statement_series(df_q, ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'], default=0).iloc[0]

            # Cash flows flow items (sum trailing 4 quarters)
            if not tail_qc.empty:
                ltm_capex = -_widest_capex(tail_qc).sum(min_count=4)
                ltm_da = _statement_series(
                    tail_qc,
                    ['Depreciation And Amortization', 'Depreciation Amortization Depletion', 'Depreciation']
                ).abs().sum(min_count=4)
                cash_flows.loc[latest_q, 'Capital Expenditure'] = ltm_capex
                cash_flows.loc[latest_q, 'Depreciation And Amortization'] = ltm_da

    except Exception:
        pass

    return income_statement, balance_sheet, cash_flows


def get_historical_data(ticker_symbol):
    """Fetch and process historical financial data, preferring SEC annual filings."""
    using_sec = True
    try:
        income_statement, balance_sheet, cash_flows = get_sec_historical_statements(ticker_symbol)
    except Exception:
        using_sec = False
        income_statement, balance_sheet, cash_flows = _get_yahoo_historical_statements(ticker_symbol)

    cash_flow_fields = {
        "Capital Expenditure": ["Capital Expenditure"],
        "Depreciation And Amortization": [
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
            "Depreciation",
        ],
    }
    missing_cash_flow_data = any(
        field not in cash_flows.columns or cash_flows[field].tail(4).notna().sum() < 2
        for field in cash_flow_fields
    )
    if missing_cash_flow_data:
        try:
            yahoo_cash_flows = yf.Ticker(ticker_symbol).cashflow.T.sort_index()
            for field, names in cash_flow_fields.items():
                if field == "Capital Expenditure":
                    yahoo_values = _reindex_nearest(-_widest_capex(yahoo_cash_flows), cash_flows.index)
                else:
                    yahoo_values = _reindex_nearest(_statement_series(yahoo_cash_flows, names), cash_flows.index)
                existing_values = _statement_series(cash_flows, [field])
                cash_flows[field] = existing_values.combine_first(yahoo_values)
        except Exception:
            pass

    # Some companies (e.g., restaurants) never tag GrossProfit/CostOfRevenue in
    # SEC XBRL since their filings don't present a distinct COGS subtotal;
    # Yahoo's reclassified statements often fill this gap. Applies to any ticker.
    income_statement_fields = {
        "Gross Profit": ["Gross Profit"],
        "Cost Of Revenue": ["Cost Of Revenue"],
    }
    missing_income_data = any(
        field not in income_statement.columns or income_statement[field].tail(4).notna().sum() < 2
        for field in income_statement_fields
    )
    if missing_income_data:
        try:
            yahoo_income_statement = yf.Ticker(ticker_symbol).financials.T.sort_index()
            for field, names in income_statement_fields.items():
                yahoo_values = _reindex_nearest(_statement_series(yahoo_income_statement, names), income_statement.index)
                existing_values = _statement_series(income_statement, [field])
                income_statement[field] = existing_values.combine_first(yahoo_values)
        except Exception:
            pass

    # Augment with the latest available quarterly LTM data.
    income_statement, balance_sheet, cash_flows = _augment_with_latest_quarterly(
        ticker_symbol, income_statement, balance_sheet, cash_flows
    )

    revenue_for_window = _statement_series(
        income_statement,
        ['Total Revenue', 'Operating Revenue'],
    ).dropna().sort_index()
    latest_year = revenue_for_window.index.max().year if not revenue_for_window.empty else pd.Timestamp.today().year
    target_years = set(range(latest_year - 5, latest_year + 1))
    available_target_periods = revenue_for_window.index[
        revenue_for_window.index.year.isin(target_years)
    ]
    if using_sec and len(available_target_periods) < 4:
        try:
            income_statement, balance_sheet, cash_flows = _get_yahoo_historical_statements(ticker_symbol)
            using_sec = False
            income_statement, balance_sheet, cash_flows = _augment_with_latest_quarterly(
                ticker_symbol, income_statement, balance_sheet, cash_flows
            )
            revenue_for_window = _statement_series(
                income_statement,
                ['Total Revenue', 'Operating Revenue'],
            ).dropna().sort_index()
            if not revenue_for_window.empty:
                latest_year = revenue_for_window.index.max().year
                target_years = set(range(latest_year - 5, latest_year + 1))
            available_target_periods = revenue_for_window.index[
                revenue_for_window.index.year.isin(target_years)
            ]
        except Exception:
            pass

    historical_periods = available_target_periods
    # Drop only near-duplicate period ends (e.g., 52/53-week calendars), not a
    # whole fiscal year that shares a calendar year with the LTM point:
    # January fiscal-year filers (NVDA, WMT) used to lose their latest 10-K.
    kept_periods = []
    for period in historical_periods.sort_values():
        if kept_periods and (period - kept_periods[-1]).days < 45:
            kept_periods[-1] = period
        else:
            kept_periods.append(period)
    historical_periods = pd.DatetimeIndex(kept_periods)
    if len(historical_periods) < 2:
        historical_periods = revenue_for_window.index[-5:]
    if len(historical_periods) < 2:
        raise ValueError(f"No sufficient annual revenue history was returned for {ticker_symbol}.")
    prior_period = revenue_for_window.index[revenue_for_window.index < historical_periods[0]][-1:]
    balance_periods = prior_period.append(historical_periods)

    rev_hist = revenue_for_window.reindex(balance_periods)
    def _comparable_period_pct_change(series):
        # A single latest LTM point cannot be compared with the preceding
        # fiscal-year total as a growth rate; the periods overlap.
        # A gap longer than a year (a missing fiscal year) is annualized so it
        # isn't averaged in as if it were one year of growth.
        gap_years = pd.Series(series.index, index=series.index).diff().dt.days / 365.25
        growth = series.pct_change()
        long = gap_years.notna() & (gap_years > 1.1) & (growth > -1)
        growth = growth.where(~long, (1 + growth) ** (1 / gap_years) - 1)
        short = gap_years.notna() & (gap_years < 0.9)
        return growth.where(~short)

    revenue_growth = _comparable_period_pct_change(rev_hist).reindex(historical_periods).replace([np.inf, -np.inf], np.nan)
    revenue_growth_change = revenue_growth.diff()
    ebit_for_growth = _statement_series(
        income_statement,
        ['Operating Income', 'EBIT'],
    ).dropna().sort_index().reindex(balance_periods)
    ebit_growth = _comparable_period_pct_change(ebit_for_growth).reindex(historical_periods).replace([np.inf, -np.inf], np.nan)

    income_statement = income_statement.reindex(historical_periods)
    balance_sheet = balance_sheet.reindex(balance_periods)
    cash_flows = cash_flows.reindex(historical_periods)

    # Normalize statement labels because Yahoo omits some lines for industries
    # such as biotech and uses alternate names for depreciation.
    income_statement['Total Revenue'] = _statement_series(
        income_statement, ['Total Revenue', 'Operating Revenue']
    )
    income_statement['EBIT'] = _statement_series(
        income_statement, ['Operating Income', 'EBIT']
    )
    if income_statement['EBIT'].isna().all():
        # Pre-interest income is the closest proxy for operating income; net
        # income (after interest and tax) is a last resort.
        net_income = _statement_series(
            income_statement,
            ['Net Income', 'Net Income Common Stockholders'],
        )
        pretax_income = _statement_series(income_statement, ['Pretax Income'])
        interest_expense = _statement_series(
            income_statement,
            ['Interest Expense', 'Interest Expense Non Operating'],
        )
        if pretax_income.notna().any() and interest_expense.notna().any():
            income_statement['EBIT'] = pretax_income + interest_expense.abs().fillna(0)
        elif pretax_income.notna().any():
            income_statement['EBIT'] = pretax_income
        else:
            income_statement['EBIT'] = net_income
    gross_profit = _statement_series(income_statement, ['Gross Profit'])
    if gross_profit.isna().all():
        cost_of_revenue = _statement_series(income_statement, ['Cost Of Revenue'])
        gross_profit = income_statement['Total Revenue'] - cost_of_revenue
    income_statement['Gross Profit'] = gross_profit
    income_statement['Gross Margin'] = (
        income_statement['Gross Profit'] / income_statement['Total Revenue']
    ).replace([np.inf, -np.inf], np.nan)
    income_statement['EBIT Margin'] = (
        income_statement['EBIT'] / income_statement['Total Revenue']
    ).replace([np.inf, -np.inf], np.nan)
    income_statement['Revenue Growth'] = revenue_growth.reindex(income_statement.index)
    income_statement['Revenue Growth Change'] = revenue_growth_change.reindex(income_statement.index)
    income_statement['EBIT Growth'] = ebit_growth.reindex(income_statement.index)
    if income_statement['EBIT Growth'].eq(0).all() and income_statement['EBIT'].notna().sum() > 1:
        income_statement['EBIT Growth'] = (
            income_statement['EBIT'].pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

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
    # Clip so a near-zero pretax year can't produce a nonsensical rate that
    # distorts NOPAT, return on capital, and reinvestment for that year.
    eff_tax_rate = eff_tax_rate.replace([np.inf, -np.inf], np.nan).fillna(0.21).clip(0, 0.5)

    # Calculate NWC from balance sheet
    cash_col = 'Cash Cash Equivalents And Short Term Investments' if 'Cash Cash Equivalents And Short Term Investments' in balance_sheet.columns else 'Cash And Cash Equivalents'
    debt_col = 'Current Debt And Capital Lease Obligation' if 'Current Debt And Capital Lease Obligation' in balance_sheet.columns else 'Current Debt'

    current_assets = _statement_series(balance_sheet, ['Current Assets'])
    current_liabilities = _statement_series(balance_sheet, ['Current Liabilities'])
    cash = _statement_series(balance_sheet, [cash_col])
    current_debt = _statement_series(balance_sheet, [debt_col])
    balance_sheet["Adj CA"] = current_assets - cash
    balance_sheet["Adj CL"] = current_liabilities - current_debt
    balance_sheet["NWC"] = balance_sheet["Adj CA"] - balance_sheet["Adj CL"]
    balance_sheet["Ch in NWC"] = balance_sheet["NWC"].diff()

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
    invested_cash = _statement_series(
        balance_sheet,
        ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents'],
        default=0,
    )
    invested_capital = total_debt + equity - invested_cash
    invested_capital = invested_capital.where(invested_capital > 0)
    average_invested_capital = invested_capital.rolling(2, min_periods=2).mean()

    balance_sheet = balance_sheet.reindex(historical_periods)

    # Process cash flows
    widest_capex = _widest_capex(cash_flows)
    cash_flows['Capital Expenditure'] = widest_capex
    cash_flows['Depreciation And Amortization'] = _statement_series(
        cash_flows,
        ['Depreciation And Amortization', 'Depreciation Amortization Depletion', 'Depreciation'],
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
    df_stats['Reinv Rate'] = (
        merged_cf['Reinvestment'] / income_statement['NOPAT']
    ).replace([np.inf, -np.inf], np.nan)

    df_stats['Return on Capital'] = (
        income_statement['NOPAT'] / average_invested_capital.reindex(income_statement.index)
    ).replace([np.inf, -np.inf], np.nan)
    # Revenue generated per dollar of invested capital; links projected
    # growth to the reinvestment needed to fund it.
    df_stats['Sales to Capital'] = (
        income_statement['Total Revenue'] / average_invested_capital.reindex(income_statement.index)
    ).replace([np.inf, -np.inf], np.nan)

    return {
        'income_statement': income_statement,
        'balance_sheet': balance_sheet,
        'cash_flows': cash_flows,
        'merged_cf': merged_cf,
        'df_stats': df_stats
    }


def get_ltm_revenue(ticker_symbol):
    """Get a reliable LTM revenue from the latest four valid quarters."""
    ticker = yf.Ticker(ticker_symbol)

    quarterly_data = None
    for statement_getter in (
        lambda: ticker.quarterly_income_stmt,
        lambda: ticker.quarterly_financials,
    ):
        try:
            statement = statement_getter()
            if statement is not None and not statement.empty:
                quarterly_data = statement.T.sort_index()
                break
        except Exception:
            continue

    if quarterly_data is None or quarterly_data.empty:
        raise ValueError(f"No quarterly financial data was returned for {ticker_symbol}.")

    revenue_column = None
    for possible_column in ("Total Revenue", "Operating Revenue", "Revenue"):
        if possible_column in quarterly_data.columns and pd.to_numeric(
            quarterly_data[possible_column], errors="coerce"
        ).notna().any():
            revenue_column = possible_column
            break

    if revenue_column is None:
        available_columns = ", ".join(map(str, quarterly_data.columns))
        raise ValueError(
            f"Yahoo Finance did not return a usable revenue line for {ticker_symbol}. "
            f"Available fields: {available_columns}"
        )

    revenue = pd.to_numeric(quarterly_data[revenue_column], errors="coerce").dropna().sort_index()
    if revenue.empty:
        raise ValueError(f"No usable revenue history was returned for {ticker_symbol}.")

    revenue = revenue.tail(4)
    if len(revenue) < 4:
        raise ValueError(
            f"Only {len(revenue)} usable quarters of revenue data were returned for {ticker_symbol}; "
            "four quarters are required for LTM revenue."
        )

    ltm_revenue = float(revenue.sum())
    most_recent_date = revenue.index[-1]

    if not np.isfinite(ltm_revenue) or ltm_revenue < 0:
        raise ValueError(f"Invalid LTM revenue returned for {ticker_symbol}.")

    return ltm_revenue, most_recent_date


def build_dcf_projections(ltm_revenue, most_recent_date, growth_rates, ebit_margins, reinv_rates, eff_tax_rate):
    """Build projections dataframe for DCF model."""
    time_horizon = len(growth_rates)
    if time_horizon == 0:
        raise ValueError("At least one projection year is required.")
    if len(ebit_margins) != time_horizon or len(reinv_rates) != time_horizon:
        raise ValueError("Each projection year requires growth, margin, and reinvestment assumptions.")
    if not all(np.isfinite(values).all() for values in (growth_rates, ebit_margins, reinv_rates)):
        raise ValueError("Projection assumptions must contain finite numbers.")
    if not np.isfinite(ltm_revenue) or ltm_revenue <= 0:
        raise ValueError("LTM revenue must be greater than zero.")
    if not np.isfinite(eff_tax_rate) or not 0 <= eff_tax_rate <= 1:
        raise ValueError("The tax rate must be between 0% and 100%.")
    if any(rate < -1 for rate in growth_rates):
        raise ValueError("Revenue growth cannot be below -100%, which would imply negative revenue.")

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
    projections['Reinvestment'] = projections['NOPAT'] * projections['Reinv Rate']
    projections['FCF'] = projections['NOPAT'] - projections['Reinvestment']
    projections['T'] = range(1, time_horizon + 1)

    if not np.isfinite(projections[['Revenue', 'EBIT', 'NOPAT', 'FCF']].to_numpy()).all():
        raise ValueError("Projection assumptions produced non-finite financial results.")

    return projections


def calculate_dcf_valuation(projections, wacc, terminal_growth, total_debt, total_cash, shares_outstanding,
                            non_operating_assets=0.0, minority_interest=0.0, fx_rate=1.0):
    """Calculate DCF valuation and implied share price.

    Uses the mid-year convention: each year's FCF is discounted from mid-year,
    since cash arrives through the year rather than on its last day, and the
    perpetuity is shifted by the same half year to stay consistent.
    Amounts are in the statement currency; `fx_rate` converts equity value to
    the quote currency for the per-share price.
    """
    time_horizon = len(projections)

    if time_horizon == 0:
        raise ValueError("At least one projection year is required.")
    if not np.isfinite(wacc) or wacc <= 0:
        raise ValueError("WACC must be greater than 0%.")
    if not np.isfinite(terminal_growth) or terminal_growth <= -1 or terminal_growth >= wacc:
        raise ValueError("Perpetual growth must be above -100% and lower than WACC.")
    if not np.isfinite(shares_outstanding) or shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be greater than zero.")
    if not np.isfinite(fx_rate) or fx_rate <= 0:
        raise ValueError("The exchange rate must be greater than zero.")
    if not all(np.isfinite(value) and value >= 0 for value in
               (total_debt, total_cash, non_operating_assets, minority_interest)):
        raise ValueError("Equity bridge amounts must be finite and non-negative.")

    # Discount FCF to present
    timing_shift = 0.5
    projections = projections.copy()
    projections['Discounted_FCF'] = projections['FCF'] / (1 + wacc) ** (projections['T'] - timing_shift)
    pv_fcf = projections['Discounted_FCF'].sum()

    # Terminal value using Gordon Growth Model
    terminal_fcf = projections.iloc[-1]['FCF'] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** (time_horizon - timing_shift)

    # Enterprise value and equity value. Non-operating investments are added
    # because their income is excluded from operating income; minority
    # interest is the outside owners' share of consolidated cash flows.
    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value - total_debt + total_cash + non_operating_assets - minority_interest
    share_price = equity_value * fx_rate / shares_outstanding

    return {
        'pv_fcf': pv_fcf,
        'terminal_value': terminal_value,
        'terminal_fcf': terminal_fcf,
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

def _update_wacc_beta():
    beta, note = select_beta(
        BETA_METHODS[st.session_state["wacc_beta_method"]],
        st.session_state["wacc_regression_beta"],
        st.session_state["wacc_bottom_up_beta"],
    )
    st.session_state["wacc_beta_input"] = float(beta)
    st.session_state["wacc_beta_note"] = note


def _update_wacc_spread():
    st.session_state["wacc_spread_input"] = get_credit_spread(
        st.session_state["wacc_credit_rating"]
    ) * 100


def render_wacc():
    st.header("WACC Calculator")
    st.caption("Market estimates are starting points. Adjust any input before applying WACC to the DCF.")
    ticker_symbol = st.text_input("Ticker Symbol", value="MSFT", key="shared_ticker_symbol").strip().upper()
    if not ticker_symbol:
        st.warning("Enter a ticker symbol.")
        return

    if st.session_state.get("wacc_defaults_ticker") != ticker_symbol:
        with st.spinner(f"Loading market assumptions for {ticker_symbol}..."):
            risk_free, premium, tax, rating, coverage = get_wacc_input_defaults(ticker_symbol)
            warnings = []
            try:
                company = get_company_market_data(ticker_symbol)
                market_cap = company["market_cap"] / 1e6
                debt = company["total_debt"] * company["fx_rate"] / 1e6
                currency = company["price_currency"]
            except Exception as error:
                market_cap, debt, currency = 0.0, 0.0, "quote currency"
                warnings.append(f"Could not load capital structure: {error}. Enter it manually.")
            try:
                raw_beta, r_squared, _, _ = calculate_beta(ticker_symbol)
                regression_beta = adjust_beta_for_wacc(raw_beta)
                if not np.isfinite(regression_beta):
                    raise ValueError("Regression beta is not finite")
                bottom_up = get_bottom_up_beta(
                    ticker_symbol, debt / market_cap if market_cap > 0 else 0, tax
                )
                beta_detail = f"Blume-adjusted regression beta {regression_beta:.3f} (R² {r_squared:.3f})."
            except Exception as error:
                regression_beta, bottom_up = 1.0, None
                beta_detail = "Placeholder beta: enter your own estimate."
                warnings.append(f"Could not estimate beta: {error}. The beta placeholder is 1.0.")

            st.session_state.update({
                "wacc_risk_free_rate": risk_free * 100,
                "wacc_emrp": premium * 100,
                "wacc_tax_rate": tax * 100,
                "wacc_credit_rating": rating,
                "wacc_spread_input": get_credit_spread(rating) * 100,
                "wacc_beta_method": next(iter(BETA_METHODS)),
                "wacc_regression_beta": regression_beta,
                "wacc_bottom_up_beta": bottom_up,
                "wacc_beta_detail": beta_detail,
                "wacc_market_cap_input": float(market_cap),
                "wacc_debt_input": float(debt),
                "wacc_currency": currency,
                "wacc_load_warnings": warnings,
                "wacc_interest_coverage": coverage,
                "wacc_defaults_ticker": ticker_symbol,
            })
            _update_wacc_beta()
    for warning in st.session_state["wacc_load_warnings"]:
        st.warning(warning)

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_free = st.number_input("Risk-Free Rate (%)", step=0.1, key="wacc_risk_free_rate") / 100
        premium = st.number_input("Equity Risk Premium (%)", step=0.1, key="wacc_emrp") / 100
        tax = st.number_input("Marginal Tax Rate (%)", min_value=0.0, max_value=100.0,
                              step=1.0, key="wacc_tax_rate") / 100
    with col2:
        st.selectbox("Beta Method", options=list(BETA_METHODS), key="wacc_beta_method",
                     on_change=_update_wacc_beta)
        beta = st.number_input("Beta", step=0.05, key="wacc_beta_input")
        st.caption(f"{st.session_state['wacc_beta_note']} {st.session_state['wacc_beta_detail']} "
                   "Edit beta directly if needed.")
        st.selectbox("Credit Rating", options=[entry["Rating"] for entry in CREDIT_SPREADS],
                     key="wacc_credit_rating", on_change=_update_wacc_spread)
        spread = st.number_input("Credit Spread (%)", step=0.1, key="wacc_spread_input") / 100
    with col3:
        currency = st.session_state["wacc_currency"]
        market_cap = st.number_input(f"Market Capitalization (millions, {currency})",
                                     min_value=0.0, step=1.0, key="wacc_market_cap_input")
        debt = st.number_input(f"Debt (millions, {currency})",
                               min_value=0.0, step=1.0, key="wacc_debt_input")
        st.caption("The industry beta is estimated using the initial capital structure and tax rate; "
                   "override beta if you change them.")

    if market_cap + debt <= 0:
        st.warning("Enter a positive market capitalization or debt to calculate WACC.")
        return
    equity_weight = market_cap / (market_cap + debt)
    debt_weight = 1 - equity_weight
    cost_of_equity = risk_free + beta * premium
    cost_of_debt = risk_free + spread
    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax)
    cols = st.columns(4)
    cols[0].metric("Cost of Equity", f"{cost_of_equity:.2%}")
    cols[1].metric("Cost of Debt", f"{cost_of_debt:.2%}")
    cols[2].metric("Equity / Debt Weights", f"{equity_weight:.1%} / {debt_weight:.1%}")
    cols[3].metric("WACC", f"{wacc:.2%}")
    if not np.isfinite(wacc) or wacc <= 0:
        st.error("WACC must be positive before it can be used in the DCF.")
    elif st.button("Use WACC in DCF", type="primary"):
        st.session_state["wacc_calculated_ticker"] = ticker_symbol
        st.session_state["wacc_calculated_rate"] = wacc
        st.session_state["dcf_wacc_rate"] = float(wacc * 100)
        st.success("WACC applied to the DCF model.")


# ============================================================================
# Page: Historical Analysis
# ============================================================================

def annual_assumption_history(stats):
    """Exclude the latest LTM observation when it overlaps the prior fiscal year."""
    stats = stats.sort_index()
    if isinstance(stats.index, pd.DatetimeIndex) and len(stats) > 1:
        if (stats.index[-1] - stats.index[-2]).days < 0.9 * 365.25:
            return stats.iloc[:-1]
    return stats


def render_historical():
    st.header("Historical Analysis")
    st.markdown("Analyze historical financial performance: growth rates, margins, and reinvestment.")

    ticker_symbol = st.session_state.get("shared_ticker_symbol", "MSFT").strip().upper()

    # Inputs
    scale_column, _, _ = st.columns(3)

    with scale_column:
        scale_options = {"Millions ($M)": 1_000_000, "Billions ($B)": 1_000_000_000}
        scale_choice = st.selectbox(
            "Display Scale", options=list(scale_options.keys()), index=0
        )
        scale_factor = scale_options[scale_choice]
        scale_name = "$M" if scale_factor == 1_000_000 else "$B"

    analyze_button = st.button("Analyze Financials", type="primary")

    saved_data = st.session_state.get("historical_analysis_data")
    saved_ticker = st.session_state.get("historical_analysis_ticker")
    has_saved_analysis = saved_data is not None and saved_ticker == ticker_symbol

    if analyze_button or has_saved_analysis:
        with st.spinner(f"Fetching financial data for {ticker_symbol}..."):
            try:
                if analyze_button:
                    ticker = yf.Ticker(ticker_symbol)
                    company_name = ticker.info.get('longName', ticker_symbol)
                    data = get_historical_data(ticker_symbol)
                    st.session_state["historical_analysis_data"] = data
                    st.session_state["historical_analysis_ticker"] = ticker_symbol
                    st.session_state["historical_analysis_company"] = company_name
                    st.session_state["dcf_projection_refresh_needed"] = True
                else:
                    data = saved_data
                    company_name = st.session_state.get("historical_analysis_company", ticker_symbol)

                st.subheader(f"{company_name} ({ticker_symbol})")

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
                st.markdown("### Historical Averages & Latest Run-rate (for projections)")
                annual_stats = annual_assumption_history(df_stats)
                st.caption("Averages use available annual observations; an overlapping latest LTM period is excluded.")

                avg_col1, avg_col2, avg_col3, avg_col4 = st.columns(4)

                with avg_col1:
                    avg_rev_growth = annual_stats['Revenue Growth'].replace([np.inf, -np.inf], np.nan).mean()
                    latest_rev_growth = df_stats['Revenue Growth'].dropna().iloc[-1] if not df_stats['Revenue Growth'].dropna().empty else avg_rev_growth
                    st.metric(
                        "Avg Revenue Growth",
                        f"{avg_rev_growth:.2%}",
                        help="Multi-year historical average revenue growth"
                    )
                    st.caption(f"**Latest available:** {latest_rev_growth:.2%}")

                with avg_col2:
                    avg_ebit_margin = annual_stats['EBIT Margin'].replace([np.inf, -np.inf], np.nan).mean()
                    latest_ebit_margin = df_stats['EBIT Margin'].dropna().iloc[-1] if not df_stats['EBIT Margin'].dropna().empty else avg_ebit_margin
                    st.metric("Avg EBIT Margin", f"{avg_ebit_margin:.2%}")
                    st.caption(f"**Latest available:** {latest_ebit_margin:.2%}")

                with avg_col3:
                    avg_tax_rate = annual_stats['Eff Tax Rate'].mean()
                    latest_tax_rate = df_stats['Eff Tax Rate'].dropna().iloc[-1] if not df_stats['Eff Tax Rate'].dropna().empty else avg_tax_rate
                    st.metric("Avg Eff Tax Rate", f"{avg_tax_rate:.2%}")
                    st.caption(f"**Latest available:** {latest_tax_rate:.2%}")

                with avg_col4:
                    avg_reinv_rate = annual_stats['Reinv Rate'].replace([np.inf, -np.inf], np.nan).mean()
                    latest_reinv_rate = df_stats['Reinv Rate'].dropna().iloc[-1] if not df_stats['Reinv Rate'].dropna().empty else avg_reinv_rate
                    st.metric("Avg Reinv Rate", f"{avg_reinv_rate:.2%}")
                    st.caption(f"**Latest available:** {latest_reinv_rate:.2%}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check that the ticker symbol is valid and has sufficient financial data.")

# ============================================================================
# Page: DCF Model
# ============================================================================

def historical_scenario_defaults(historical_data):
    """Use annual historical means for the starting base assumptions."""
    stats = annual_assumption_history(historical_data["df_stats"]).replace([np.inf, -np.inf], np.nan)
    values = {}
    missing = []
    for column in ("Revenue Growth", "EBIT Margin", "Reinv Rate"):
        series = pd.to_numeric(stats[column], errors="coerce").dropna()
        if series.empty:
            missing.append(column)
            values[column] = 0.0
        else:
            values[column] = float(series.mean()) * 100
    return values, missing


def build_scenario_assumptions(growth, margin, reinvestment, margin_change):
    """Apply a symmetric relative margin shock; all inputs are percentages."""
    values = (growth, margin, reinvestment, margin_change)
    if not all(np.isfinite(value) for value in values) or margin_change < 0:
        raise ValueError("Scenario assumptions must be finite and the margin change cannot be negative.")
    margin_delta = abs(margin) * margin_change / 100
    return pd.DataFrame({
        "Scenario": ["Bear", "Base", "Best"],
        "Revenue growth (%)": [growth] * 3,
        "EBIT margin (%)": [margin - margin_delta, margin, margin + margin_delta],
        "Reinvestment rate (%)": [reinvestment] * 3,
    })


def scenario_range_warning(results):
    """Flag inverted or substantially uneven scenario values without changing them."""
    bear, base, best = (results[name]["share_price"] for name in ("Bear", "Base", "Best"))
    if not all(np.isfinite(value) for value in (bear, base, best)):
        return "Scenario values are not finite. Review the assumptions."
    if bear > base or base > best:
        return "Scenario values are not ordered Bear ≤ Base ≤ Best. Review the operating assumptions."
    downside, upside = base - bear, best - base
    if max(downside, upside) > 0 and min(downside, upside) / max(downside, upside) < 0.8:
        return ("The upside and downside ranges differ by more than 20%. "
                "Review the scenario assumptions; DCF values need not be symmetric.")
    return None


def render_dcf():
    st.header("DCF Model")
    st.caption("Project free cash flow to the firm and discount it at WACC. "
               "Review all fetched figures and assumptions before relying on the result.")
    ticker = st.session_state.get("shared_ticker_symbol", "MSFT").strip().upper()
    if not ticker:
        st.warning("Enter a ticker in the WACC Calculator.")
        return

    if st.session_state.get("dcf_loaded_ticker") != ticker or st.session_state.get("dcf_projection_refresh_needed", False):
        warnings = []
        with st.spinner(f"Loading financial data for {ticker}..."):
            try:
                if st.session_state.get("historical_analysis_ticker") == ticker:
                    history = st.session_state["historical_analysis_data"]
                else:
                    history = get_historical_data(ticker)
                defaults, missing = historical_scenario_defaults(history)
                if missing:
                    warnings.append(f"Historical data missing for {', '.join(missing)}; "
                                    "zero placeholders require your review.")
            except Exception as error:
                defaults = {"Revenue Growth": 0.0, "EBIT Margin": 0.0, "Reinv Rate": 0.0}
                warnings.append(f"Could not load historical assumptions: {error}. Enter them manually.")
            try:
                revenue, revenue_date = get_ltm_revenue(ticker)
            except Exception as error:
                revenue, revenue_date = 0.0, pd.Timestamp.today().normalize()
                warnings.append(f"Could not load LTM revenue: {error}. Enter revenue and its date manually.")
            try:
                company = get_company_market_data(ticker)
            except Exception as error:
                company = {
                    "company_name": ticker, "financial_currency": "statement currency",
                    "price_currency": "quote currency", "fx_rate": 1.0,
                    "diluted_shares_outstanding": 0.0, "total_debt": 0.0, "total_cash": 0.0,
                    "non_operating_assets": 0.0, "minority_interest": 0.0, "current_price": 0.0,
                }
                warnings.append(f"Could not load the equity bridge: {error}. Enter the figures manually.")
            st.session_state.update({
                "dcf_loaded_ticker": ticker,
                "dcf_projection_refresh_needed": False,
                "dcf_company": company,
                "dcf_load_warnings": warnings,
                "dcf_scenario_revision": st.session_state.get("dcf_scenario_revision", 0) + 1,
                "dcf_base_growth": defaults["Revenue Growth"],
                "dcf_base_margin": defaults["EBIT Margin"],
                "dcf_base_reinvestment": defaults["Reinv Rate"],
                "dcf_margin_change": 10.0,
                "dcf_custom_scenarios": False,
                "dcf_revenue": float(revenue / 1e6),
                "dcf_revenue_date": pd.Timestamp(revenue_date).date(),
                "dcf_tax_rate": float(get_effective_tax_rate(ticker) * 100),
                "dcf_debt": float(company["total_debt"] / 1e6),
                "dcf_cash": float(company["total_cash"] / 1e6),
                "dcf_investments": float(company["non_operating_assets"] / 1e6),
                "dcf_minority": float(company["minority_interest"] / 1e6),
                "dcf_shares": float(company["diluted_shares_outstanding"] / 1e6),
                "dcf_fx": float(company["fx_rate"]),
                "dcf_price": float(company["current_price"] or 0.0),
            })
            if st.session_state.get("wacc_calculated_ticker") == ticker:
                st.session_state["dcf_wacc_rate"] = float(st.session_state["wacc_calculated_rate"] * 100)
            else:
                st.session_state["dcf_wacc_rate"] = 9.0
                warnings.append("WACC is a 9% placeholder; calculate or enter a rate.")
            st.session_state["dcf_terminal_growth"] = 3.0

    company = st.session_state["dcf_company"]
    for warning in st.session_state["dcf_load_warnings"]:
        st.warning(warning)
    st.subheader(f"{company['company_name']} ({ticker})")
    st.caption(f"Financial amounts are in millions of {company['financial_currency']}; "
               f"share prices are in {company['price_currency']}. Verify debt, cash and investments "
               "do not overlap and that shares reflect dilution.")

    st.markdown("### Valuation inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        revenue = st.number_input("LTM Revenue (millions)", min_value=0.0, step=1.0, key="dcf_revenue") * 1e6
        revenue_date = st.date_input("LTM End Date", key="dcf_revenue_date")
        years = st.number_input("Forecast Years", min_value=1, max_value=30, value=5, step=1, key="dcf_years")
    with c2:
        wacc = st.number_input("WACC (%)", step=0.25, key="dcf_wacc_rate") / 100
        terminal_growth = st.number_input("Perpetual Growth (%)", step=0.25, key="dcf_terminal_growth") / 100
        tax = st.number_input("Operating Tax Rate (%)", min_value=0.0, max_value=100.0,
                              step=1.0, key="dcf_tax_rate") / 100
    with c3:
        shares = st.number_input("Diluted Shares (millions)", min_value=0.0, step=1.0, key="dcf_shares") * 1e6
        fx = st.number_input("FX: Quote Currency per Statement Currency", min_value=0.0,
                             step=0.01, format="%.6f", key="dcf_fx")
        price = st.number_input("Current Share Price", min_value=0.0, step=0.1, key="dcf_price")

    with st.expander("Equity value bridge", expanded=True):
        b1, b2, b3, b4 = st.columns(4)
        debt = b1.number_input("Debt (millions)", min_value=0.0, step=1.0, key="dcf_debt") * 1e6
        cash = b2.number_input("Cash (millions)", min_value=0.0, step=1.0, key="dcf_cash") * 1e6
        investments = b3.number_input("Non-operating Investments (millions)", min_value=0.0,
                                      step=1.0, key="dcf_investments") * 1e6
        minority = b4.number_input("Minority Interest (millions)", min_value=0.0,
                                   step=1.0, key="dcf_minority") * 1e6

    st.markdown("### Scenario assumptions")
    st.caption("Base assumptions start at the annual historical averages and remain constant through the forecast. "
               "An overlapping LTM period is excluded. Bear and best apply equal relative changes to the base EBIT margin; "
               "growth, reinvestment, WACC and perpetual growth stay the same.")
    a1, a2, a3, a4 = st.columns(4)
    growth = a1.number_input("Base Revenue Growth (%)", step=0.5, key="dcf_base_growth")
    margin = a2.number_input("Base EBIT Margin (%)", step=0.5, key="dcf_base_margin")
    reinvestment = a3.number_input("Base Reinvestment Rate (%)", step=0.5, key="dcf_base_reinvestment")
    margin_change = a4.number_input(
        "EBIT Margin Change (±% of base)", min_value=0.0, step=1.0, key="dcf_margin_change",
        help="For a 20% base margin and a 10% change, bear is 18% and best is 22%.",
    )
    scenarios = build_scenario_assumptions(growth, margin, reinvestment, margin_change)
    if margin == 0:
        st.warning("The base EBIT margin is zero, so a relative margin change produces identical cases. "
                   "Enter a margin or use custom scenarios.")
    custom = st.checkbox("Edit individual scenario assumptions", key="dcf_custom_scenarios")
    if custom:
        st.caption("Custom inputs can produce unequal ranges; use assumptions consistent with each case.")
        scenarios = st.data_editor(
            scenarios,
            key=f"dcf_scenarios_{st.session_state['dcf_scenario_revision']}",
            hide_index=True,
            disabled=["Scenario"],
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Revenue growth (%)": st.column_config.NumberColumn(format="%.2f", required=True),
                "EBIT margin (%)": st.column_config.NumberColumn(format="%.2f", required=True),
                "Reinvestment rate (%)": st.column_config.NumberColumn(format="%.2f", required=True),
            },
        )
    else:
        st.dataframe(scenarios.style.format({
            "Revenue growth (%)": "{:.2f}", "EBIT margin (%)": "{:.2f}",
            "Reinvestment rate (%)": "{:.2f}",
        }), hide_index=True, use_container_width=True)
    try:
        results = {}
        for _, case in scenarios.iterrows():
            growth = float(case["Revenue growth (%)"]) / 100
            margin = float(case["EBIT margin (%)"]) / 100
            reinvestment = float(case["Reinvestment rate (%)"]) / 100
            projections = build_dcf_projections(
                revenue, pd.Timestamp(revenue_date), [growth] * years,
                [margin] * years, [reinvestment] * years, tax
            )
            results[case["Scenario"]] = calculate_dcf_valuation(
                projections, wacc, terminal_growth, debt, cash, shares,
                non_operating_assets=investments, minority_interest=minority, fx_rate=fx
            )
    except (ValueError, TypeError) as error:
        st.error(f"Cannot calculate DCF: {error}")
        return

    warning = scenario_range_warning(results)
    if warning:
        st.warning(warning)
    st.markdown("### Implied value per share")
    for column, name in zip(st.columns(3), ("Bear", "Base", "Best")):
        implied = results[name]["share_price"]
        delta = f"{(implied / price - 1):+.1%} vs. market" if price > 0 else None
        column.metric(name, f"{implied:,.2f} {company['price_currency']}", delta)
    st.caption("Default cases are an operating-margin sensitivity, not probability-weighted forecasts. "
               "A symmetrical input change does not guarantee symmetrical values when other scenario inputs differ.")
    if price <= 0:
        st.caption("Enter a positive current share price to see upside or downside.")
    if any(result["terminal_fcf"] < 0 for result in results.values()):
        st.warning("At least one scenario has negative terminal free cash flow. Review whether a perpetual-growth DCF is appropriate.")
    st.caption("Terminal value = final-year FCF × (1 + perpetual growth) / (WACC − perpetual growth). "
               "The final-year reinvestment rate therefore continues in perpetuity. "
               "Check that perpetual reinvestment is sufficient to support growth. "
               "Cash flows and terminal value use the mid-year discount convention.")

    selected = st.selectbox("Show details for", ["Base", "Bear", "Best"])
    valuation = results[selected]
    st.markdown("### Forecast and valuation")
    forecast = valuation["projections"][["Revenue", "EBIT", "NOPAT", "Reinvestment", "FCF", "Discounted_FCF"]] / 1e6
    forecast.index = forecast.index.strftime("%Y-%m-%d")
    st.dataframe(forecast.style.format("{:,.2f}"), use_container_width=True)
    st.caption(f"Amounts in millions of {company['financial_currency']}.")
    bridge = pd.DataFrame({
        "Metric": ["PV of forecast FCF", "PV of terminal value", "Enterprise value",
                   "Less debt", "Add cash", "Add non-operating investments",
                   "Less minority interest", "Equity value"],
        "Value (millions)": [valuation["pv_fcf"] / 1e6, valuation["pv_terminal"] / 1e6,
                             valuation["enterprise_value"] / 1e6, -debt / 1e6, cash / 1e6,
                             investments / 1e6, -minority / 1e6, valuation["equity_value"] / 1e6],
    })
    st.dataframe(bridge.style.format({"Value (millions)": "{:,.2f}"}), hide_index=True, use_container_width=True)


# ============================================================================
# Main App with Tabs
# ============================================================================

if __name__ == "__main__":
    inject_styles()
    st.markdown("<div class='brand-mark'>PATEL DCF</div>", unsafe_allow_html=True)
    tab_home, tab_wacc, tab_historical, tab_dcf = st.tabs([
        "🏠 Home", "1️⃣ WACC Calculator", "2️⃣ Historical Analysis", "3️⃣ DCF Model"
    ])
    with tab_home:
        render_home()
    with tab_wacc:
        render_wacc()
    with tab_historical:
        render_historical()
    with tab_dcf:
        render_dcf()
