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

## Architecture

The application is based on three analytical modules (originally Colab notebooks):

1. **WACC Module** (`dcf1_wacc_inclass_0945.py`): Computes weighted average cost of capital
   - Equity/debt weights from market cap and total debt
   - Cost of equity via CAPM (beta from OLS regression on 5-year monthly returns)
   - Cost of debt via risk-free rate + credit spread lookup

2. **Historical Analysis Module** (`dcf2_historical_analysis_inclass_0945.py`): Analyzes firm financials
   - Revenue/EBIT growth rates and margins from income statement
   - NWC changes from balance sheet
   - Reinvestment rate from cash flow statement

3. **DCF Model Module** (`dcf3_dcf_model_inclass_0945.py`): Computes valuation
   - Projects revenue, EBIT, NOPAT, FCF based on growth/margin/reinvestment assumptions
   - Discounts FCF to present value
   - Terminal value via Gordon Growth Model
   - Derives implied share price

## External APIs

- **yfinance**: Stock data, financials, market cap, debt
- **FRED API**: Risk-free rate (10-year Treasury yield) - requires API key

## Key Financial Formulas

- WACC = w_E × k_E + w_D × k_D × (1-t)
- Cost of Equity (CAPM): k_E = r_f + β × EMRP
- Cost of Debt: k_D = r_f + credit_spread
- FCF = NOPAT × (1 - Reinvestment Rate)
- Terminal Value = FCF_final × (1 + g) / (WACC - g)
