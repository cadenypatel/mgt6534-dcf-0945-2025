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
        .stButton > button[kind="primary"] p { color: #ffffff; }
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

# The banking convention for the ERP: Kroll's recommended US equity risk
# premium, paired with the spot 10-year Treasury while it is above Kroll's
# 3.5% normalized rate. 5.0% since Kroll's June 2024 update.
RECOMMENDED_ERP = 0.050

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


@st.cache_data(ttl=900, show_spinner=False)
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
    # Yahoo's share count can cover a single class (GOOGL, BRK-B, NVO) while
    # its market cap covers all classes; market cap ÷ price counts every class
    # in units of the quoted share.
    if market_cap is not None and current_price is not None:
        shares_outstanding = max(shares_outstanding or 0.0, market_cap / current_price)

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
    reported_diluted_shares = None
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
                reported_diluted_shares = float(diluted[latest_date])
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
        "reported_diluted_shares": reported_diluted_shares,
        "total_debt": total_debt if total_debt is not None else 0,
        "total_cash": total_cash if total_cash is not None else 0,
        "non_operating_assets": non_operating_assets,
        "minority_interest": minority_interest,
        "current_price": current_price,
        "sector": ticker_info.get("sector") or "",
        "industry": ticker_info.get("industry") or "",
        "ev_to_ebitda": _positive_number(ticker_info.get("enterpriseToEbitda")),
    }


def shares_jumped(company):
    """True when shares outstanding exceed the last reported diluted count by
    more than 15%: new shares issued after the latest statements, usually for
    an acquisition that has closed. (Foreign ADRs are skipped: their reported
    count is in local shares.)"""
    reported = company.get("reported_diluted_shares")
    return bool(reported and company["fx_rate"] == 1.0 and company["diluted_shares_outstanding"] > 1.15 * reported)


def compute_wacc(ticker_symbol, risk_free_rate, emrp, marginal_tax_rate, firm_rating, beta_method):
    """WACC from market-value weights, CAPM cost of equity, and a rating-based
    cost of debt. `beta_method` is a BETA_METHODS value."""
    company_data = get_company_market_data(ticker_symbol)
    market_cap = company_data["market_cap"]
    # Debt is reported in the statement currency; market cap is in the quote currency.
    total_debt = company_data["total_debt"] * company_data["fx_rate"]
    w_E = market_cap / (market_cap + total_debt)
    w_D = total_debt / (market_cap + total_debt)

    raw_beta, r_squared, _, _ = calculate_beta(ticker_symbol)
    regression_beta = adjust_beta_for_wacc(raw_beta)
    bottom_up = get_bottom_up_beta(ticker_symbol, total_debt / market_cap, marginal_tax_rate)
    beta, beta_note = select_beta(beta_method, regression_beta, bottom_up)

    cost_of_equity = risk_free_rate + beta * emrp
    credit_spread = get_credit_spread(firm_rating)
    cost_of_debt = risk_free_rate + credit_spread
    wacc = (w_E * cost_of_equity) + (w_D * cost_of_debt * (1 - marginal_tax_rate))
    return {
        "company_name": company_data["company_name"],
        "market_cap": market_cap, "total_debt": total_debt, "w_E": w_E, "w_D": w_D,
        "raw_beta": raw_beta, "r_squared": r_squared, "regression_beta": regression_beta,
        "bottom_up": bottom_up, "beta": beta, "beta_note": beta_note,
        "cost_of_equity": cost_of_equity, "credit_spread": credit_spread, "cost_of_debt": cost_of_debt,
        "wacc": wacc,
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

    # EMRP is a market-wide assumption, not a company-specific input; the
    # Damodaran implied ERP is shown beside it as an alternative.
    equity_market_risk_premium = RECOMMENDED_ERP

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
    """Normalized effective tax rate: the median over the historical years with
    positive pretax income, so one year's one-off charge or benefit doesn't
    set the rate forever. A company with no profitable year pays the marginal
    rate once it earns money (loss carryforwards are not modeled). Falls back
    to Yahoo's latest rate, then 21%."""
    try:
        historical = get_historical_data(ticker_symbol)
        pretax = pd.to_numeric(historical["income_statement"].get("Pretax Income"), errors="coerce")
        tax_rates = historical["df_stats"]["Eff Tax Rate"].replace([np.inf, -np.inf], np.nan)
        profitable = tax_rates[(pretax > 0).reindex(tax_rates.index, fill_value=False)].dropna()
        usable_tax_rates = profitable[(profitable >= 0) & (profitable <= 0.5)]
        if not usable_tax_rates.empty:
            return float(usable_tax_rates.median())
        if pretax is not None and pretax.notna().any():
            return MARGINAL_TAX_RATE
    except Exception:
        pass
    try:
        tax_rate = _non_negative_number(yf.Ticker(ticker_symbol).info.get("taxRateForCalcs"))
        if tax_rate is not None and tax_rate <= 0.5:
            return tax_rate
    except Exception:
        pass
    return 0.21


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
        if take_max and not combined.empty:
            # NaN-aware max: a tag missing a year must not erase another tag's value.
            combined = pd.concat([combined, current], axis=1).max(axis=1)
        else:
            combined = combined.combine_first(current)

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
    short_term_investments = _sec_fact_series(
        facts,
        ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
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
    # Short-term investments are cash-like: leaving them in current assets
    # overstates working capital and invested capital.
    balance_sheet["Cash Cash Equivalents And Short Term Investments"] = (
        balance_sheet["Cash And Cash Equivalents"] + short_term_investments.reindex(balance_index).fillna(0)
    )
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
    """Add a trailing-twelve-month (LTM) period from the latest four quarters
    when they run past the latest annual filing."""
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

        # A quarter within ~6 weeks of the fiscal year-end is that year's 10-K period.
        if len(rev_quarterly) >= 4 and (annual_max is None or (latest_q - annual_max).days > 45):
            tail_qi = qi.loc[:latest_q].tail(4)
            if (_statement_series(tail_qi, ['Total Revenue', 'Operating Revenue']).notna().sum() < 4
                    or (tail_qi.index[-1] - tail_qi.index[0]).days > 300):
                return income_statement, balance_sheet, cash_flows
            tail_qc = qc.loc[:latest_q].tail(4) if qc is not None and not qc.empty else pd.DataFrame()
            latest_qb = qb.loc[:latest_q].iloc[-1] if qb is not None and not qb.empty else pd.Series()
            df_q = pd.DataFrame([latest_qb]) if not latest_qb.empty else pd.DataFrame()

            # Income statement flow items (sum trailing 4 quarters)
            income_statement.loc[latest_q, 'Total Revenue'] = _statement_series(tail_qi, ['Total Revenue', 'Operating Revenue']).sum()
            # Yahoo's quarterly "EBIT" includes non-operating income (investment
            # gains, equity-method income); operating income is the FCFF driver.
            # A quarter whose operating income exceeds pretax income + interest
            # by more than 15% of revenue is a data error; use the rebuild. (The
            # reverse is normal: investment gains lift pretax income.)
            quarter_operating = _statement_series(tail_qi, ['Operating Income', 'EBIT'])
            quarter_rebuilt = (_statement_series(tail_qi, ['Pretax Income'])
                               + _statement_series(tail_qi, ['Interest Expense', 'Interest Expense Non Operating']).abs().fillna(0))
            quarter_revenue = _statement_series(tail_qi, ['Total Revenue', 'Operating Revenue']).abs()
            inconsistent = (quarter_operating - quarter_rebuilt) > 0.15 * quarter_revenue
            ltm_operating_income = quarter_operating.where(~inconsistent, quarter_rebuilt).sum()
            income_statement.loc[latest_q, 'EBIT'] = ltm_operating_income
            if 'Operating Income' in income_statement.columns:
                income_statement.loc[latest_q, 'Operating Income'] = ltm_operating_income
            income_statement.loc[latest_q, 'Gross Profit'] = _statement_series(tail_qi, ['Gross Profit']).sum()
            income_statement.loc[latest_q, 'Cost Of Revenue'] = _statement_series(tail_qi, ['Cost Of Revenue']).sum()
            income_statement.loc[latest_q, 'Tax Provision'] = _statement_series(tail_qi, ['Tax Provision']).sum()
            income_statement.loc[latest_q, 'Pretax Income'] = _statement_series(tail_qi, ['Pretax Income']).sum()
            income_statement.loc[latest_q, 'Net Income'] = _statement_series(tail_qi, ['Net Income', 'Net Income Common Stockholders']).sum()
            income_statement.loc[latest_q, 'Interest Expense'] = _statement_series(tail_qi, ['Interest Expense', 'Interest Expense Non Operating']).sum()

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
                ltm_capex = -_widest_capex(tail_qc).sum()
                ltm_da = _statement_series(
                    tail_qc,
                    ['Depreciation And Amortization', 'Depreciation Amortization Depletion', 'Depreciation']
                ).abs().sum()
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
        # A revenue tag that covers only part of the business shows up as
        # operating income larger than revenue; Yahoo's statements are whole.
        sec_margins = (income_statement['EBIT'] / income_statement['Total Revenue']).replace([np.inf, -np.inf], np.nan)
        if (sec_margins.tail(6).abs() > 1.5).any():
            raise ValueError("SEC revenue tag looks partial")
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
        field not in cash_flows.columns or cash_flows[field].tail(5).isna().any()
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
    # Some filers (e.g., pharma) never tag an operating-income subtotal either.
    income_statement_fields = {
        "Gross Profit": ["Gross Profit"],
        "Cost Of Revenue": ["Cost Of Revenue"],
        "EBIT": ["Operating Income"],
    }
    missing_income_data = any(
        field not in income_statement.columns or income_statement[field].tail(5).isna().any()
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

    income_statement, balance_sheet, cash_flows = _augment_with_latest_quarterly(
        ticker_symbol, income_statement, balance_sheet, cash_flows
    )

    revenue_for_window = _statement_series(
        income_statement,
        ['Total Revenue', 'Operating Revenue'],
    ).dropna().sort_index()
    # Five fiscal years plus the LTM period: a full cycle for the averages.
    current_year = pd.Timestamp.today().year
    target_years = set(range(current_year - 5, current_year + 1))
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
    if income_statement['EBIT'].isna().any():
        # Year by year: pre-interest income is the closest proxy for operating
        # income; net income (after interest and tax) is a last resort.
        net_income = _statement_series(
            income_statement,
            ['Net Income', 'Net Income Common Stockholders'],
        )
        pretax_income = _statement_series(income_statement, ['Pretax Income'])
        interest_expense = _statement_series(
            income_statement,
            ['Interest Expense', 'Interest Expense Non Operating'],
        )
        income_statement['EBIT'] = (
            income_statement['EBIT']
            .fillna(pretax_income + interest_expense.abs().fillna(0))
            .fillna(net_income)
        )
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

    current_assets = _statement_series(balance_sheet, ['Current Assets'], default=0)
    current_liabilities = _statement_series(balance_sheet, ['Current Liabilities'], default=0)
    cash = _statement_series(balance_sheet, [cash_col], default=0)
    current_debt = _statement_series(balance_sheet, [debt_col], default=0)
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
    cash_flows['Capital Expenditure'] = widest_capex.fillna(0) if widest_capex.isna().all() else widest_capex
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

    df_stats['Return on Capital'] = (
        income_statement['NOPAT'] / average_invested_capital.reindex(income_statement.index)
    ).replace([np.inf, -np.inf], np.nan)
    # Unlevered free cash flow drivers, as a share of revenue.
    revenue = income_statement['Total Revenue']
    df_stats['D&A % Revenue'] = (merged_cf['Depreciation And Amortization'].abs() / revenue).replace([np.inf, -np.inf], np.nan)
    df_stats['Capex % Revenue'] = (merged_cf['Capital Expenditure'].abs() / revenue).replace([np.inf, -np.inf], np.nan)
    df_stats['NWC % Revenue'] = (merged_cf['NWC'] / revenue).replace([np.inf, -np.inf], np.nan)

    return {
        'income_statement': income_statement,
        'balance_sheet': balance_sheet,
        'cash_flows': cash_flows,
        'merged_cf': merged_cf,
        'df_stats': df_stats
    }


# ============================================================================
# DCF: default assumptions, projections, valuation
# ============================================================================

# Default perpetual growth: long-run nominal GDP / inflation, the standard
# 2-3% range used in practice.
PERPETUAL_GROWTH = 0.025

# Perpetual growth stays at least this far below WACC; a smaller gap makes
# the terminal value swamp everything else.
MIN_TERMINAL_SPREAD = 0.03


def default_perpetual_growth(wacc=None):
    """Perpetual growth = PERPETUAL_GROWTH, kept at least MIN_TERMINAL_SPREAD
    below WACC."""
    growth = PERPETUAL_GROWTH
    if wacc is not None and np.isfinite(wacc):
        growth = min(growth, wacc - MIN_TERMINAL_SPREAD)
    return max(growth, 0.0)


def _fiscal_year_remaining(ticker, as_of):
    """Share of the current fiscal year still ahead of `as_of`: the weight on
    current-year ("0y") consensus when blending it with next year's ("+1y")
    into the twelve months after `as_of`."""
    last_fy_end = ticker.info.get("lastFiscalYearEnd")
    if not last_fy_end:
        return 0.5
    fy_end = pd.Timestamp(last_fy_end, unit="s")
    as_of = pd.Timestamp(as_of).tz_localize(None)
    while fy_end <= as_of:
        fy_end += pd.DateOffset(years=1)
    return float(np.clip((fy_end - as_of).days / 365.25, 0.0, 1.0))


def get_consensus_growth(ticker_symbol, ltm_revenue, ltm_date):
    """Analyst-consensus revenue growth for the twelve months after the LTM
    date (current and next fiscal-year consensus, blended by how much of the
    current fiscal year is still ahead), or None if there is no consensus."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        revenue = pd.to_numeric(ticker.revenue_estimate["avg"], errors="coerce")
        revenue_fy0, revenue_fy1 = _positive_number(revenue.get("0y")), _positive_number(revenue.get("+1y"))
        if revenue_fy0 is None or revenue_fy1 is None or not ltm_revenue or ltm_revenue <= 0:
            return None
        remaining = _fiscal_year_remaining(ticker, ltm_date)
        growth = (remaining * revenue_fy0 + (1 - remaining) * revenue_fy1) / ltm_revenue - 1
        return float(growth) if np.isfinite(growth) else None
    except Exception:
        return None


def _annual_rows(stats):
    """Drop a trailing LTM period that overlaps the last fiscal year, so its
    months aren't counted twice in historical averages."""
    if len(stats) >= 2 and isinstance(stats.index, pd.DatetimeIndex):
        if (stats.index[-1] - stats.index[-2]).days < 0.9 * 365.25:
            return stats.iloc[:-1]
    return stats


def _historical_mean(stats, column):
    values = pd.to_numeric(_annual_rows(stats)[column], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _latest(stats, column):
    values = pd.to_numeric(stats[column], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan


def get_dcf_defaults(historical_data, ltm_revenue, consensus_growth=None, deal_closed=False):
    """Default DCF inputs, the same rule for every company:

    - Starting revenue: LTM revenue.
    - Revenue growth: analyst consensus for the next twelve months (else the
      historical average), stepping down in a straight line to perpetual
      growth by the final year.
    - EBIT margin, D&A, capex and net working capital: historical averages,
      % of revenue, held flat (averages span the investment cycle).

    Consensus far outside the company's history (more than 20 points beyond
    its best or worst year) reflects an acquisition or divestiture: a one-time
    step, not growth. The step goes into the starting revenue (pro forma), in
    full if the deal has closed (`deal_closed`: its shares are already
    outstanding), otherwise averaged with historical growth; growth then
    starts from the historical average.
    """
    stats = historical_data['df_stats'].replace([np.inf, -np.inf], np.nan)
    defaults = {'Revenue': float(ltm_revenue), 'notes': [], 'missing': []}

    def pick(*values, fallback, label):
        for value in values:
            if value is not None and np.isfinite(value):
                return float(value)
        defaults['missing'].append(label)
        return fallback

    average_growth = pick(_historical_mean(stats, 'Revenue Growth'), fallback=0.03, label='revenue growth')
    history = pd.to_numeric(_annual_rows(stats)['Revenue Growth'], errors='coerce').dropna()
    if consensus_growth is None:
        defaults['Revenue Growth'] = average_growth
        defaults['growth_source'] = 'historical average (no analyst consensus)'
    elif len(history) and not history.min() - 0.20 <= consensus_growth <= history.max() + 0.20:
        step = consensus_growth if deal_closed else (consensus_growth + average_growth) / 2
        defaults['Revenue'] = float(ltm_revenue * (1 + step))
        defaults['Revenue Growth'] = average_growth
        defaults['growth_source'] = 'historical average'
        defaults['notes'].append(
            f"Consensus implies {consensus_growth:.0%} revenue growth vs {history.min():.0%} to {history.max():.0%} "
            "historically, which reflects an acquisition or divestiture rather than growth. Starting revenue is "
            f"pro forma: LTM stepped up {step:.0%} ("
            + ("the full step, since the deal has closed" if deal_closed else "averaged with history, as the deal is pending")
            + "). Growth then starts from the historical average."
        )
    else:
        defaults['Revenue Growth'] = float(consensus_growth)
        defaults['growth_source'] = 'analyst consensus for the next twelve months'

    defaults['EBIT Margin'] = pick(_historical_mean(stats, 'EBIT Margin'), _latest(stats, 'EBIT Margin'),
                                   fallback=0.15, label='EBIT margin')
    defaults['D&A % Revenue'] = pick(_historical_mean(stats, 'D&A % Revenue'), _latest(stats, 'D&A % Revenue'),
                                     fallback=0.04, label='D&A')
    defaults['Capex % Revenue'] = pick(_historical_mean(stats, 'Capex % Revenue'), _latest(stats, 'Capex % Revenue'),
                                       fallback=defaults['D&A % Revenue'], label='capex')
    defaults['NWC % Revenue'] = pick(_historical_mean(stats, 'NWC % Revenue'), fallback=0.0,
                                     label='working capital')
    return defaults


def growth_path(start, terminal, horizon):
    """Year-by-year revenue growth: `start` in year 1, moving in a straight
    line to `terminal` (perpetual growth) in the final year, so the forecast
    hands off smoothly to the terminal value."""
    if horizon == 1:
        return [float(start)]
    return [start + (terminal - start) * year / (horizon - 1) for year in range(horizon)]


@st.cache_data(ttl=900, show_spinner=False)
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
    if len(revenue) < 4 or (revenue.index[-1] - revenue.index[0]).days > 300:
        # A missing quarter would stretch "LTM" over five quarters; start
        # from the latest full fiscal year instead.
        try:
            annual = pd.to_numeric(ticker.financials.T.sort_index()["Total Revenue"], errors="coerce").dropna()
            if not annual.empty and annual.iloc[-1] > 0:
                return float(annual.iloc[-1]), annual.index[-1]
        except Exception:
            pass
        raise ValueError(
            f"Yahoo did not return four consecutive quarters or an annual revenue figure for {ticker_symbol}."
        )

    ltm_revenue = float(revenue.sum())
    most_recent_date = revenue.index[-1]

    if not np.isfinite(ltm_revenue) or ltm_revenue < 0:
        raise ValueError(f"Invalid LTM revenue returned for {ticker_symbol}.")

    return ltm_revenue, most_recent_date


def build_projections(revenue, start_date, growth_rates, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct):
    """Year-by-year unlevered free cash flow:

    Revenue → EBIT = revenue × margin → NOPAT = EBIT × (1 − tax)
    Unlevered FCF = NOPAT + D&A − capex − increase in NWC
    Margin, D&A, capex and NWC are flat percentages of revenue.
    """
    horizon = len(growth_rates)
    if horizon == 0:
        raise ValueError("At least one projection year is required.")
    if not np.isfinite(growth_rates).all():
        raise ValueError("Revenue growth must be a finite number.")
    if not all(np.isfinite(value) for value in (revenue, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct)):
        raise ValueError("Revenue, margin, tax, D&A, capex and working capital must be finite numbers.")
    if revenue <= 0:
        raise ValueError("Starting revenue must be greater than zero.")
    if any(rate <= -1 for rate in growth_rates):
        raise ValueError("Revenue growth cannot be -100% or lower.")

    year_end = {month: f"YE-{name}" for month, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), start=1)}
    dates = pd.date_range(start=start_date, periods=horizon + 1, freq=year_end[pd.Timestamp(start_date).month])[1:]

    revenues = revenue * np.cumprod(1 + np.asarray(growth_rates, dtype=float))
    prior = np.concatenate([[revenue], revenues[:-1]])
    projections = pd.DataFrame(index=dates)
    projections['Revenue Growth'] = growth_rates
    projections['Revenue'] = revenues
    projections['EBIT'] = revenues * ebit_margin
    projections['Taxes'] = projections['EBIT'] * tax_rate
    projections['NOPAT'] = projections['EBIT'] - projections['Taxes']
    projections['D&A'] = revenues * da_pct
    projections['Capex'] = revenues * capex_pct
    projections['Change in NWC'] = (revenues - prior) * nwc_pct
    projections['Unlevered FCF'] = (projections['NOPAT'] + projections['D&A']
                                    - projections['Capex'] - projections['Change in NWC'])
    projections['Year'] = range(1, horizon + 1)
    return projections


def value_projections(projections, wacc, terminal_growth, bridge):
    """Enterprise value, equity value and price per share.

    Mid-year convention: each year's cash flow is discounted from the middle
    of the year. Terminal value (Gordon growth) = final-year FCF × (1 + g) ÷
    (WACC − g), discounted on the same basis.
    """
    if not np.isfinite(wacc) or wacc <= 0:
        raise ValueError("WACC must be greater than 0%.")
    if not np.isfinite(terminal_growth) or terminal_growth >= wacc:
        raise ValueError("Perpetual growth must be lower than WACC.")
    if not np.isfinite(bridge['shares']) or bridge['shares'] <= 0:
        raise ValueError("Shares outstanding must be greater than zero.")

    horizon = len(projections)
    projections = projections.copy()
    projections['Discount Factor'] = 1 / (1 + wacc) ** (projections['Year'] - 0.5)
    projections['PV of FCF'] = projections['Unlevered FCF'] * projections['Discount Factor']
    pv_fcf = projections['PV of FCF'].sum()
    terminal_value = projections['Unlevered FCF'].iloc[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** (horizon - 0.5)
    enterprise_value = pv_fcf + pv_terminal
    equity_value = (enterprise_value - bridge['total_debt'] + bridge['total_cash']
                    + bridge['non_operating_assets'] - bridge['minority_interest'])
    return {
        'projections': projections,
        'pv_fcf': pv_fcf,
        'terminal_value': terminal_value,
        'pv_terminal': pv_terminal,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'share_price': equity_value * bridge['fx_rate'] / bridge['shares'],
    }


def run_dcf(case, wacc, terminal_growth, growth_shift=0.0, margin_shift=0.0):
    """Value one case; the shifts move starting growth and the margin (scenarios)."""
    growth_rates = growth_path(case['growth'] + growth_shift, terminal_growth, case['horizon'])
    projections = build_projections(case['revenue'], case['start_date'], growth_rates, case['ebit_margin'] + margin_shift,
                                    case['tax_rate'], case['da_pct'], case['capex_pct'], case['nwc_pct'])
    return value_projections(projections, wacc, terminal_growth, case)


def current_ev_to_ebitda(company, historical_data):
    """Today's EV / LTM EBITDA, built in the statement currency from market cap,
    debt, cash and the LTM statements (Yahoo's ratio mixes currencies for
    foreign filers). None when EBITDA isn't positive or the multiple is above
    50x: near-zero EBITDA makes it meaningless as an exit multiple."""
    try:
        income = historical_data['income_statement']
        ebitda = float(income['EBIT'].dropna().iloc[-1]
                       + historical_data['merged_cf']['Depreciation And Amortization'].dropna().iloc[-1])
        enterprise_value = (company['market_cap'] / company['fx_rate'] + company['total_debt']
                            - company['total_cash'] + company['minority_interest'] - company['non_operating_assets'])
        multiple = enterprise_value / ebitda if ebitda > 0 and enterprise_value > 0 else None
        return multiple if multiple is not None and multiple <= 50 else None
    except Exception:
        return None


def exit_multiple_price(valuation, case, wacc, exit_multiple):
    """Share price with the terminal value at an EV/EBITDA exit multiple
    instead of perpetual growth; an exit is a sale at the end of the final
    year, so it is discounted a full N years."""
    final = valuation['projections'].iloc[-1]
    final_ebitda = final['EBIT'] + final['D&A']
    pv_terminal = final_ebitda * exit_multiple / (1 + wacc) ** case['horizon']
    equity_value = (valuation['pv_fcf'] + pv_terminal - case['total_debt'] + case['total_cash']
                    + case['non_operating_assets'] - case['minority_interest'])
    return equity_value * case['fx_rate'] / case['shares'], final_ebitda


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
        ("02 / Evidence", "Historical Analysis", "Review growth, margins, capex, D&A and working capital."),
        ("03 / Value", "DCF Model", "Project free cash flow, discount it at WACC, and test scenarios and sensitivities."),
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
        st.markdown("**Unlevered Free Cash Flow**")
        st.latex(r"UFCF = EBIT(1-t) + D\&A - CapEx - \Delta NWC")

        st.markdown("**Terminal Value**")
        st.latex(r"TV = \frac{UFCF_{final} \times (1 + g)}{WACC - g}")

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
        ticker_symbol = st.text_input(
            "Ticker Symbol", value="MSFT", key="shared_ticker_symbol"
        ).strip().upper()

    if st.session_state.get("wacc_defaults_ticker") != ticker_symbol:
        with st.spinner(f"Loading market assumptions for {ticker_symbol}..."):
            risk_free_default, emrp_default, tax_default, rating_default, ic_ratio = get_wacc_input_defaults(ticker_symbol)
        st.session_state["wacc_risk_free_rate"] = risk_free_default * 100
        st.session_state["wacc_emrp"] = emrp_default * 100
        st.session_state["wacc_tax_rate"] = tax_default * 100
        st.session_state["wacc_credit_rating"] = rating_default
        st.session_state["wacc_interest_coverage"] = ic_ratio
        st.session_state["wacc_defaults_ticker"] = ticker_symbol

    with col1:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0, max_value=20.0, step=0.1, key="wacc_risk_free_rate",
            help="Auto-populated from the latest 10-year Treasury yield (^TNX)"
        ) / 100

    with col2:
        emrp = st.number_input(
            "Equity Market Risk Premium (%)",
            min_value=0.0, max_value=20.0, step=0.1, key="wacc_emrp",
            help="Defaults to Kroll's recommended US ERP (5.0%), the convention in banking valuations. "
                 "Damodaran's implied ERP is the market-implied alternative."
        ) / 100
        implied_erp, implied_erp_date = get_damodaran_erp()
        st.caption(f"Kroll recommended 5.0%. Damodaran implied ERP: {implied_erp:.2%} ({implied_erp_date}).")
        rating_options = [entry["Rating"] for entry in CREDIT_SPREADS]
        firm_rating = st.selectbox(
            "Credit Rating", options=rating_options, key="wacc_credit_rating",
            help="Estimated from the ticker's interest coverage ratio (EBIT / Interest Expense); this is a modeled synthetic rating, not the company's official agency rating, so override it if you know the actual rating"
        )
        ic_ratio = st.session_state.get("wacc_interest_coverage")
        if ic_ratio is not None:
            st.caption(f"Estimated from interest coverage of {ic_ratio:.1f}x; verify against the company's actual published rating if known.")

    with col3:
        marg_tax_rate = st.number_input(
            "Marginal Tax Rate (%)",
            min_value=0.0, max_value=50.0, step=1.0, key="wacc_tax_rate",
            help="Interest is deducted at the marginal rate, so the debt tax shield uses it "
                 "(US default: 21% federal + ~4% state = 25%) rather than the effective rate"
        ) / 100
        beta_method = st.selectbox(
            "Beta Method",
            options=list(BETA_METHODS),
            key="wacc_beta_method",
            help="Blend (default): average of the company's Blume-adjusted regression beta and Damodaran's "
                 "bottom-up industry beta. Averaging two noisy estimates beats either alone; in a 10-company "
                 "test it had the lowest bias and error versus market prices.",
        )

    st.caption("Risk-free rate, tax rate, and credit rating refresh when the ticker changes. EMRP is a market-wide assumption and remains editable.")

    calculate_button = st.button("Calculate WACC", type="primary")

    if calculate_button:
        with st.spinner(f"Fetching data for {ticker_symbol}..."):
            try:
                result = compute_wacc(
                    ticker_symbol, risk_free_rate, emrp, marg_tax_rate, firm_rating, BETA_METHODS[beta_method]
                )
                (market_cap, total_debt, w_E, w_D, raw_beta, r_squared, regression_beta, bottom_up, beta,
                 beta_note, cost_of_equity, credit_spread, cost_of_debt, wacc) = (
                    result[key] for key in (
                        "market_cap", "total_debt", "w_E", "w_D", "raw_beta", "r_squared", "regression_beta",
                        "bottom_up", "beta", "beta_note", "cost_of_equity", "credit_spread", "cost_of_debt", "wacc",
                    )
                )

                st.subheader(f"{result['company_name']} ({ticker_symbol})")
                st.session_state["wacc_calculated_ticker"] = ticker_symbol
                st.session_state["wacc_calculated_rate"] = wacc
                st.session_state["wacc_cost_of_equity"] = cost_of_equity

                st.markdown(
                    "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:.5rem 0 1rem'>"
                    + kpi_card("WACC", f"{wacc:.2%}", "Loaded into the DCF tab")
                    + kpi_card("Cost of Equity", f"{cost_of_equity:.2%}", f"{risk_free_rate:.2%} + {beta:.2f} × {emrp:.2%}")
                    + kpi_card("After-Tax Cost of Debt", f"{cost_of_debt * (1 - marg_tax_rate):.2%}",
                               f"({risk_free_rate:.2%} + {credit_spread:.2%} {firm_rating}) × (1 − {marg_tax_rate:.0%})")
                    + kpi_card("Weights", f"{w_E:.0%} / {w_D:.0%}", "Equity / debt at market value")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Beta {beta:.3f}: {beta_note}")
                st.caption(
                    f"Regression: raw {raw_beta:.3f} → Blume-adjusted {regression_beta:.3f} "
                    f"(R² {r_squared:.3f}, 5-year monthly vs S&P 500)"
                    + (
                        f" · Bottom-up: {bottom_up[1]} unlevered {bottom_up[2]:.2f} relevered at D/E "
                        f"{total_debt / market_cap:.1%} → {bottom_up[0]:.3f}"
                        if bottom_up else ""
                    )
                )
                st.caption(
                    f"Market cap \\${market_cap / 1e9:,.1f}B · total debt \\${total_debt / 1e9:,.1f}B · "
                    "spread from Damodaran's latest rating table."
                )
                st.session_state["dcf_wacc_rate"] = wacc * 100

                st.session_state["dcf_projection_refresh_needed"] = True
                try:
                    historical_data = get_historical_data(ticker_symbol)
                    st.session_state["historical_analysis_data"] = historical_data
                    st.session_state["historical_analysis_ticker"] = ticker_symbol
                except Exception as history_error:
                    st.warning(f"WACC calculated, but historicals could not be loaded for the DCF: {history_error}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# Page: Historical Analysis
# ============================================================================

def render_historical():
    st.header("Historical Analysis")
    st.markdown("The operating history behind the DCF's default assumptions.")

    ticker_symbol = st.session_state.get("shared_ticker_symbol", "MSFT").strip().upper()
    analyze_button = st.button("Analyze Financials", type="primary")
    has_saved = (st.session_state.get("historical_analysis_data") is not None
                 and st.session_state.get("historical_analysis_ticker") == ticker_symbol)
    if not (analyze_button or has_saved):
        return

    with st.spinner(f"Fetching financial data for {ticker_symbol}..."):
        try:
            if analyze_button:
                data = get_historical_data(ticker_symbol)
                st.session_state.update({
                    "historical_analysis_data": data,
                    "historical_analysis_ticker": ticker_symbol,
                    "historical_analysis_company": get_company_market_data(ticker_symbol)["company_name"],
                    "dcf_projection_refresh_needed": True,
                })
            data = st.session_state["historical_analysis_data"]
        except Exception as error:
            st.error(f"Error: {error}")
            return

    st.subheader(f"{st.session_state.get('historical_analysis_company', ticker_symbol)} ({ticker_symbol})")
    stats = data['df_stats']
    percent_rows = ['Revenue Growth', 'EBIT Margin', 'Eff Tax Rate', 'D&A % Revenue', 'Capex % Revenue',
                    'NWC % Revenue', 'Return on Capital']
    table = data['income_statement'][['Total Revenue', 'EBIT']].div(1e6).join(stats[percent_rows])
    table = table.rename(columns={'Total Revenue': 'Revenue ($M)', 'EBIT': 'EBIT ($M)', 'Eff Tax Rate': 'Tax Rate'})
    table.index = table.index.strftime('%b %Y')
    table = table.T
    percent_rows = [row if row != 'Eff Tax Rate' else 'Tax Rate' for row in percent_rows]
    st.dataframe(
        table.style.format("{:,.0f}", na_rep="—", subset=pd.IndexSlice[['Revenue ($M)', 'EBIT ($M)'], :])
             .format("{:.1%}", na_rep="—", subset=pd.IndexSlice[percent_rows, :]),
        width="stretch",
    )
    st.caption("The last column is the latest twelve months when it runs past the last fiscal year. Working "
               "capital excludes cash and short-term investments.")

    st.markdown("### Carried into the DCF")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average EBIT Margin", f"{_historical_mean(stats, 'EBIT Margin'):.1%}")
    c2.metric("Average D&A % Revenue", f"{_historical_mean(stats, 'D&A % Revenue'):.1%}")
    c3.metric("Average Capex % Revenue", f"{_historical_mean(stats, 'Capex % Revenue'):.1%}")
    c4.metric("Average NWC % Revenue", f"{_historical_mean(stats, 'NWC % Revenue'):.1%}")
    st.caption("Revenue growth comes from analyst consensus for the next twelve months when available "
               f"(historical average: {_historical_mean(stats, 'Revenue Growth'):.1%}).")


# ============================================================================
# Page: DCF Model
# ============================================================================

def kpi_card(label, value, note=""):
    note_html = f"<p>{note}</p>" if note else ""
    return f"<div class='stat-card'><div class='label'>{label}</div><div class='value'>{value}</div>{note_html}</div>"


def load_dcf_defaults(ticker_symbol):
    """Fill the DCF inputs for a new ticker (or after the WACC tab reruns)."""
    if st.session_state.get("historical_analysis_ticker") == ticker_symbol and \
            st.session_state.get("historical_analysis_data") is not None:
        historical_data = st.session_state["historical_analysis_data"]
    else:
        historical_data = get_historical_data(ticker_symbol)
    company = get_company_market_data(ticker_symbol)
    ltm_revenue, ltm_date = get_ltm_revenue(ticker_symbol)
    consensus = get_consensus_growth(ticker_symbol, ltm_revenue, ltm_date)
    defaults = get_dcf_defaults(historical_data, ltm_revenue, consensus, shares_jumped(company))
    if st.session_state.get("wacc_calculated_ticker") == ticker_symbol:
        wacc = st.session_state["wacc_calculated_rate"]
    else:
        wacc = 0.09
        defaults['notes'].append("WACC is a 9% placeholder; run the WACC Calculator for this ticker.")
    if shares_jumped(company):
        defaults['notes'].append(
            f"Shares outstanding ({company['diluted_shares_outstanding'] / 1e6:,.0f}M) are well above the last reported "
            f"count ({company['reported_diluted_shares'] / 1e6:,.0f}M), likely from an acquisition that closed after the "
            "latest statements; their debt and cash are pre-deal. Treat the result as an estimate until the next filing."
        )
    st.session_state.update({
        "dcf_loaded_ticker": ticker_symbol,
        "dcf_projection_refresh_needed": False,
        "dcf_revenue": round(defaults['Revenue'] / 1e6, 1),
        "dcf_start_date": ltm_date,
        "dcf_growth": round(defaults['Revenue Growth'] * 100, 2),
        "dcf_margin": round(defaults['EBIT Margin'] * 100, 2),
        "dcf_da": round(defaults['D&A % Revenue'] * 100, 2),
        "dcf_capex": round(defaults['Capex % Revenue'] * 100, 2),
        "dcf_nwc": round(defaults['NWC % Revenue'] * 100, 2),
        "dcf_tax": round(get_effective_tax_rate(ticker_symbol) * 100, 2),
        "dcf_wacc_rate": round(wacc * 100, 2),
        "dcf_terminal_growth_rate": round(default_perpetual_growth(wacc) * 100, 2),
        "dcf_growth_source": defaults['growth_source'],
        "dcf_notes": defaults['notes'],
        "dcf_missing": defaults['missing'],
        "dcf_current_multiple": current_ev_to_ebitda(company, historical_data),
    })
    st.session_state.pop("dcf_run_ticker", None)


def render_dcf():
    st.header("DCF Model")
    st.caption(
        "Unlevered free cash flow = EBIT × (1 − tax) + D&A − capex − increase in working capital, discounted at "
        "WACC (mid-year). Terminal value = final-year free cash flow × (1 + g) ÷ (WACC − g)."
    )
    ticker_symbol = st.session_state.get("shared_ticker_symbol", "MSFT").strip().upper()
    if (st.session_state.get("dcf_loaded_ticker") != ticker_symbol
            or st.session_state.get("dcf_projection_refresh_needed", False)):
        with st.spinner(f"Loading assumptions for {ticker_symbol}..."):
            try:
                load_dcf_defaults(ticker_symbol)
            except Exception as error:
                st.error(f"Could not load data for {ticker_symbol}: {error}")
                return

    st.markdown(f"### Assumptions · {ticker_symbol}")
    c1, c2, c3, c4 = st.columns(4)
    wacc = c1.number_input("WACC (%)", min_value=1.0, max_value=30.0, step=0.25, key="dcf_wacc_rate",
                           help="From the WACC Calculator tab.") / 100
    terminal_growth = c2.number_input("Perpetual Growth (%)", step=0.25, key="dcf_terminal_growth_rate",
                                      help="Growth after the forecast, forever. Default 2.5% (long-run nominal "
                                           "GDP / inflation), kept at least 3 points below WACC.") / 100
    tax_rate = c3.number_input("Tax Rate (%)", min_value=0.0, max_value=50.0, step=1.0, key="dcf_tax",
                               help="Median effective tax rate over the company's profitable years.") / 100
    horizon = int(c4.selectbox("Forecast Years", options=["10", "5"], key="dcf_years"))

    c1, c2, c3 = st.columns(3)
    revenue = c1.number_input("Starting Revenue ($M)", min_value=0.0, step=100.0, key="dcf_revenue",
                              help="Latest twelve months (pro forma when consensus shows an acquisition).") * 1e6
    growth = c2.number_input("Revenue Growth (%)", step=0.5, key="dcf_growth",
                             help="Year-1 growth; steps down in a straight line to perpetual growth by the final "
                                  "year.") / 100
    ebit_margin = c3.number_input("EBIT Margin (%)", step=0.5, key="dcf_margin",
                                  help="Held flat every year. Default: historical average.") / 100
    c1, c2, c3 = st.columns(3)
    da_pct = c1.number_input("D&A (% of revenue)", min_value=0.0, step=0.25, key="dcf_da",
                             help="Held flat. Default: historical average.") / 100
    capex_pct = c2.number_input("Capex (% of revenue)", min_value=0.0, step=0.25, key="dcf_capex",
                                help="Held flat. Default: historical average.") / 100
    nwc_pct = c3.number_input("Net Working Capital (% of revenue)", step=0.5, key="dcf_nwc",
                              help="Increase in NWC = this % × increase in revenue. Default: historical average "
                                   "(excludes cash and short-term investments).") / 100
    st.caption(f"Revenue growth: {st.session_state['dcf_growth_source']}. EBIT margin, D&A, capex and working "
               "capital: historical averages.")
    for note in st.session_state.get("dcf_notes", []):
        st.warning(note)
    if st.session_state.get("dcf_missing"):
        st.warning(f"No history for {', '.join(st.session_state['dcf_missing'])}; placeholders are shown.")

    with st.expander("Scenarios & exit multiple"):
        s1, s2, s3 = st.columns(3)
        growth_swing = s1.number_input("Growth Swing (± pts)", min_value=0.0, value=3.0, step=0.5,
                                       key="dcf_growth_swing") / 100
        margin_swing = s2.number_input("Margin Swing (± pts)", min_value=0.0, value=2.0, step=0.5,
                                       key="dcf_margin_swing") / 100
        exit_multiple = s3.number_input("Exit EV/EBITDA (0 = today's multiple)", min_value=0.0, value=0.0,
                                        step=0.5, key="dcf_exit_multiple") or st.session_state.get("dcf_current_multiple")

    if terminal_growth >= wacc:
        st.error("Perpetual growth must be below WACC.")
        return
    if st.button("Run DCF Valuation", type="primary"):
        st.session_state["dcf_run_ticker"] = ticker_symbol
    if st.session_state.get("dcf_run_ticker") != ticker_symbol:
        return

    try:
        company = get_company_market_data(ticker_symbol)
    except Exception as error:
        st.error(f"Error: {error}")
        return
    price = company["current_price"]
    case = {
        'revenue': revenue, 'start_date': st.session_state["dcf_start_date"], 'horizon': horizon,
        'growth': growth, 'ebit_margin': ebit_margin, 'tax_rate': tax_rate,
        'da_pct': da_pct, 'capex_pct': capex_pct, 'nwc_pct': nwc_pct,
        'total_debt': company['total_debt'], 'total_cash': company['total_cash'],
        'non_operating_assets': company['non_operating_assets'], 'minority_interest': company['minority_interest'],
        'shares': company['diluted_shares_outstanding'], 'fx_rate': company['fx_rate'],
    }
    try:
        scenarios = {name: run_dcf(case, wacc, terminal_growth, sign * growth_swing, sign * margin_swing)
                     for name, sign in (("Bear", -1), ("Base", 0), ("Bull", 1))}
    except ValueError as error:
        st.error(str(error))
        return
    base = scenarios['Base']

    st.divider()
    st.subheader(f"{company['company_name']} ({ticker_symbol})")
    if company["financial_currency"] != company["price_currency"]:
        st.caption(f"Statements in {company['financial_currency']}, price in {company['price_currency']} "
                   f"(converted at {company['fx_rate']:.4f}).")

    def vs_market(value):
        change = value / price - 1
        color = "var(--teal)" if change >= 0 else "var(--coral)"
        return f"<span style='color:{color}'>{change:+.1%}</span> vs market"

    columns = st.columns(4)
    columns[0].markdown(kpi_card("Market Price", f"${price:,.2f}"), unsafe_allow_html=True)
    for column, name in zip(columns[1:], ("Bear", "Base", "Bull")):
        value = scenarios[name]['share_price']
        column.markdown(kpi_card(f"{name} Case", f"${value:,.2f}", vs_market(value)), unsafe_allow_html=True)
    st.caption(f"Bear / bull: revenue growth ±{growth_swing:.1%} and EBIT margin ±{margin_swing:.1%}.")

    tab_projections, tab_bridge, tab_sensitivity = st.tabs(["Projections", "Value Bridge", "Sensitivity"])
    with tab_projections:
        table = base['projections'].drop(columns=['Year']).copy()
        money = ['Revenue', 'EBIT', 'Taxes', 'NOPAT', 'D&A', 'Capex', 'Change in NWC', 'Unlevered FCF', 'PV of FCF']
        table[money] = table[money] / 1e6
        table[['Taxes', 'Capex', 'Change in NWC']] = -table[['Taxes', 'Capex', 'Change in NWC']]
        table = table.rename(columns={'Taxes': '(−) Taxes', 'D&A': '(+) D&A', 'Capex': '(−) Capex',
                                      'Change in NWC': '(−) Increase in NWC'})
        table.index = table.index.strftime('FY%y')
        table = table.T
        st.dataframe(
            table.style.format("{:.1%}", subset=pd.IndexSlice[['Revenue Growth'], :])
                 .format("{:.3f}", subset=pd.IndexSlice[['Discount Factor'], :])
                 .format("{:,.0f}", subset=pd.IndexSlice[[row for row in table.index
                                                          if row not in ('Revenue Growth', 'Discount Factor')], :]),
            width="stretch",
        )
        st.caption("$M. Each year's FCF is discounted from mid-year.")
    with tab_bridge:
        bridge = pd.DataFrame({
            "Item": ["PV of forecast FCF", "PV of terminal value", "Enterprise value", "Less: debt", "Plus: cash",
                     "Plus: non-operating investments", "Less: minority interest", "Equity value",
                     "÷ Diluted shares", "Value per share"],
            "Value": [f"{base['pv_fcf'] / 1e6:,.0f}", f"{base['pv_terminal'] / 1e6:,.0f}",
                      f"{base['enterprise_value'] / 1e6:,.0f}", f"({case['total_debt'] / 1e6:,.0f})",
                      f"{case['total_cash'] / 1e6:,.0f}", f"{case['non_operating_assets'] / 1e6:,.0f}",
                      f"({case['minority_interest'] / 1e6:,.0f})", f"{base['equity_value'] / 1e6:,.0f}",
                      f"{case['shares'] / 1e6:,.1f}M", f"${base['share_price']:,.2f}"],
        })
        st.table(bridge.set_index("Item"))
        share = base['pv_terminal'] / base['enterprise_value'] if base['enterprise_value'] > 0 else np.nan
        st.caption("$M. " + (f"Terminal value is {share:.0%} of enterprise value." if np.isfinite(share) else ""))
        if exit_multiple:
            exit_price, _ = exit_multiple_price(base, case, wacc, exit_multiple)
            st.markdown(kpi_card(f"Cross-check: exit at {exit_multiple:.1f}x EV/EBITDA", f"${exit_price:,.2f}",
                                 "Terminal value set at an EBITDA multiple instead of perpetual growth."),
                        unsafe_allow_html=True)
    with tab_sensitivity:
        st.markdown("**Value per share: WACC × perpetual growth**")
        grid = {}
        for g in (terminal_growth + step for step in (-0.01, -0.005, 0.0, 0.005, 0.01)):
            grid[f"g {g:.2%}"] = {
                f"{w:.2%}": (run_dcf(case, w, g)['share_price'] if g < w else np.nan)
                for w in (wacc + step for step in (-0.01, -0.005, 0.0, 0.005, 0.01))
            }
        grid_df = pd.DataFrame(grid)
        grid_df.index.name = "WACC"
        st.table(grid_df.style.format("${:,.2f}", na_rep="n/a")
                 .map(lambda value: "" if pd.isna(value) else
                      f"background-color: {'#d7eeeb' if value >= price else '#fbe3dd'}"))
        st.caption(f"Teal above, coral below the market price of ${price:,.2f}. Center cell is the base case.")


# ============================================================================
# Main App with Tabs
# ============================================================================

if __name__ == "__main__":
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
