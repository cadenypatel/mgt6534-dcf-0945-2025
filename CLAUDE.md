# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit web application for DCF (Discounted Cash Flow) equity valuation. The app allows users to value publicly traded companies by computing:
- WACC (Weighted Average Cost of Capital)
- Historical financial analysis (growth rates, margins, reinvestment)
- DCF model with projected cash flows and terminal value

## Tech Stack

- **Framework**: Streamlit
- **Language**: Python 3.x
- **Key Dependencies**: pandas, numpy, yfinance, statsmodels, matplotlib, fredapi

## Development Commands

```bash
# Run the Streamlit app locally
streamlit run app.py

# Install dependencies
pip install -r requirements.txt
```

## Deployment

- **GitHub Repository**: https://github.com/cadenypatel/mgt6534-dcf-0945-2025
- **Streamlit Cloud**: Deploy via https://share.streamlit.io (select repo, branch: main, file: app.py)

## App Structure

The app (`app.py`) uses a tabbed interface with four sections:

### Home Tab
- Landing page with overview and instructions
- Key formulas displayed (WACC, FCF, Terminal Value)

### Tab 1: WACC Calculator
- Inputs: Ticker symbol, risk-free rate, EMRP, credit rating, tax rate
- Beta: average of Blume-adjusted OLS regression (5-year monthly vs S&P 500) and Damodaran bottom-up industry beta (selectable)
- ERP, industry betas, and credit spreads fetched live from Damodaran's site (cached daily, stored fallbacks)
- Computes cost of equity (CAPM), cost of debt, and WACC
- Displays capital structure weights and summary table

### Tab 2: Historical Analysis
- Annual SEC statements (Yahoo fallback) plus LTM; drivers as % of revenue and the defaults carried into the DCF

### Tab 3: DCF Model
- Standard unlevered DCF, one method for every company (no sector-specific models)
- Inputs: WACC, perpetual growth, tax, forecast years, starting revenue, revenue growth (steps down linearly to perpetual growth), and flat EBIT margin, D&A, capex and NWC % of revenue
- Outputs: bear/base/bull, projections, value bridge with exit-multiple cross-check, WACC × g grid

## Reference Modules

The original Colab notebooks are preserved as reference:

1. **`dcf1_wacc_inclass_0945.py`**: WACC computation logic
2. **`dcf2_historical_analysis_inclass_0945.py`**: Historical financial analysis
3. **`dcf3_dcf_model_inclass_0945.py`**: DCF model and valuation

## Key Helper Functions in app.py

- `get_historical_data(ticker)`: statements and driver ratios (`df_stats`)
- `get_dcf_defaults(...)`: default inputs (consensus growth, historical-average ratios, pro forma revenue for deals)
- `growth_path`, `build_projections`, `value_projections`, `run_dcf`: the DCF itself
- `exit_multiple_price`, `current_ev_to_ebitda`: exit-multiple cross-check

## External APIs

- **yfinance**: Stock data, financials, market cap, debt, shares outstanding
- **FRED API**: Risk-free rate (10-year Treasury yield) - requires API key (not currently used in app)

## Key Financial Formulas

- WACC = w_E × k_E + w_D × k_D × (1-t)
- Cost of Equity (CAPM): k_E = r_f + β × EMRP
- Cost of Debt: k_D = r_f + credit_spread
- NOPAT = EBIT × (1 - Tax Rate)
- Unlevered FCF = NOPAT + D&A − Capex − ΔNWC (flat % of revenue)
- Perpetual growth default = risk-free rate, at least 3 pts below WACC; ERP default = Kroll 5.0%
- Terminal Value = UFCF_final × (1 + g) / (WACC − g), mid-year discounting
- Enterprise Value = PV(FCFs) + PV(Terminal Value)
- Equity Value = Enterprise Value - Debt + Cash + Non-operating Investments - Minority Interest
- Share Price = Equity Value × FX (statement → quote currency) / Diluted Shares

## Session History

### December 4, 2025
- Re-architected app from single WACC calculator to tabbed multi-page app
- Implemented Home landing page with overview and formulas
- Integrated Historical Analysis module (Tab 2) with full financial statement analysis
- Built complete DCF Model (Tab 3) with:
  - LTM revenue starting point
  - Customizable projection assumptions (growth, margins, reinvestment)
  - Terminal value calculation
  - Enterprise/equity value waterfall
  - Implied share price vs current market price
  - Sensitivity analysis for WACC scenarios
  - Full valuation summary table
- Pushed code to GitHub: cadenypatel/mgt6534-dcf-0945-2025
- Prepared for Streamlit Cloud deployment

### 2026 Valuation Update
- Expanded historical analysis target window from 2021-2025 to 2021-2026.
- Added `_augment_with_latest_quarterly` helper to ingest 2026 quarterly performance (Q1 & Q2):
  - Trailing 4-quarter aggregation for flow metrics (Revenue, EBIT, Gross Profit, Pretax/Net Income, Tax, CapEx, D&A) through the latest 2026 quarter to produce a consistent 12-month run-rate.
  - Point-in-time balance sheet items (Cash, Current Assets/Liabilities, Debt, Equity) mapped to latest 2026 reported quarter.
  - Dynamically aligned column discrepancies between SEC XBRL and Yahoo statements.
- Historical Analysis (Tab 2) displays 2026 metrics, NWC changes, and AI infrastructure reinvestment surges.
- DCF Model (Tab 3) automatically reflects updated 2026 historical averages for revenue growth, margins, and reinvestment rate defaults.

### November 2026: Simplified standard DCF
- Rebuilt the DCF as a plain standard model: one growth rate stepping down to perpetual growth, flat ratios, Gordon terminal value on final-year FCF
- Removed sector-specific models, return-on-capital terminal rules, steady-state capex, non-GAAP consensus conversion, industry margins, and the legacy outlier/recency weighting helpers
- Kept data fixes: SEC short-term investments as cash, capex across tags, missing operating income filled, partial SEC revenue fallback, dual-class share counts (max of Yahoo shares and market cap ÷ price), consecutive-quarter LTM, one-sided operating-income scrub, statement-based EV/EBITDA
- `calibrate.py` runs tickers sequentially (yfinance mixes data across threads)

