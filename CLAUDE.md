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

- **GitHub Repository**: https://github.com/jblocher/mgt6534-dcf-0945-2025
- **Streamlit Cloud**: Deploy via https://share.streamlit.io (select repo, branch: main, file: app.py)

## App Structure

The app (`app.py`) uses a tabbed interface with four sections:

### Home Tab
- Landing page with overview and instructions
- Key formulas displayed (WACC, FCF, Terminal Value)

### Tab 1: WACC Calculator
- Inputs: Ticker symbol, risk-free rate, EMRP, credit rating, tax rate
- Calculates beta via OLS regression (5-year monthly returns vs S&P 500)
- Computes cost of equity (CAPM), cost of debt, and WACC
- Displays capital structure weights and summary table

### Tab 2: Historical Analysis
- Inputs: Ticker symbol, display scale
- Fetches income statement, balance sheet, cash flow statement from yfinance
- Computes and displays:
  - Revenue/EBIT growth rates
  - Gross margin and EBIT margin
  - NWC and change in NWC
  - Reinvestment (CapEx - D&A + Change in NWC)
  - NOPAT and reinvestment rate
- Shows historical averages for use in projections

### Tab 3: DCF Model
- Inputs: Ticker, WACC, terminal growth rate, effective tax rate
- Projection assumptions: Revenue growth, EBIT margins, reinvestment rates (comma-separated per year)
- Fetches LTM revenue from quarterly data as starting point
- Projects Revenue → EBIT → NOPAT → FCF for each year
- Calculates terminal value via Gordon Growth Model
- Discounts all cash flows to present value
- Computes enterprise value, equity value, and implied share price
- Compares to current market price (upside/downside %)
- Includes sensitivity analysis for different WACC scenarios

## Reference Modules

The original Colab notebooks are preserved as reference:

1. **`dcf1_wacc_inclass_0945.py`**: WACC computation logic
2. **`dcf2_historical_analysis_inclass_0945.py`**: Historical financial analysis
3. **`dcf3_dcf_model_inclass_0945.py`**: DCF model and valuation

## Key Helper Functions in app.py

- `get_credit_spread(rating)`: Looks up credit spread from Damodaran table
- `calculate_beta(ticker_symbol)`: OLS regression for beta estimation
- `get_historical_data(ticker_symbol)`: Fetches and processes all financial statements
- `get_ltm_revenue(ticker_symbol)`: Gets LTM revenue from quarterly data
- `build_dcf_projections(...)`: Creates projections dataframe with all calculations
- `calculate_dcf_valuation(...)`: Computes PV of FCFs, terminal value, and share price

## External APIs

- **yfinance**: Stock data, financials, market cap, debt, shares outstanding
- **FRED API**: Risk-free rate (10-year Treasury yield) - requires API key (not currently used in app)

## Key Financial Formulas

- WACC = w_E × k_E + w_D × k_D × (1-t)
- Cost of Equity (CAPM): k_E = r_f + β × EMRP
- Cost of Debt: k_D = r_f + credit_spread
- NOPAT = EBIT × (1 - Tax Rate)
- FCF = NOPAT × (1 - Reinvestment Rate)
- Terminal Value = FCF_final × (1 + g) / (WACC - g)
- Enterprise Value = PV(FCFs) + PV(Terminal Value)
- Equity Value = Enterprise Value - Debt + Cash
- Share Price = Equity Value / Shares Outstanding

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
- Pushed code to GitHub: jblocher/mgt6534-dcf-0945-2025
- Prepared for Streamlit Cloud deployment
