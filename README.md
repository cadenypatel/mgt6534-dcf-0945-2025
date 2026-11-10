# DCF Equity Valuation App

A Streamlit web application for Discounted Cash Flow (DCF) equity valuation. Value any publicly traded company by computing WACC, analyzing historical financials, and building a complete DCF model.



## Live Demo

Deploy your own instance via [Streamlit Cloud](https://share.streamlit.io)

## Features

### WACC Calculator
- Beta (default): average of the Blume-adjusted OLS regression beta (5-year monthly returns vs S&P 500) and Damodaran's bottom-up industry beta relevered at the firm's D/E; either can be selected alone
- Equity risk premium defaults to Kroll's recommended 5.0% (the banking convention); Damodaran's live implied ERP is shown as the alternative
- Industry betas and rating spreads are fetched live from Damodaran's latest datasets (cached daily, with stored fallbacks)
- Cost of equity using CAPM (risk-free rate + beta × equity market risk premium)
- Cost of debt using risk-free rate + credit spread (Damodaran lookup table)
- Weighted average based on market cap and total debt

### Historical Analysis
- Annual statements from SEC filings (Yahoo Finance as fallback), plus the latest twelve months
- Revenue, growth, EBIT margin, tax rate, D&A, capex and working capital as a share of revenue, and return on capital

### DCF Model
A standard unlevered DCF, the same for every company:
- Starting revenue: latest twelve months
- Revenue growth: one rate (default: analyst consensus for the next twelve months, else the historical average), stepping down in a straight line to perpetual growth by the final year
- EBIT margin, D&A, capex and net working capital: one flat % of revenue each (default: historical averages)
- Unlevered FCF = EBIT × (1 − tax) + D&A − capex − increase in NWC, discounted at WACC (mid-year)
- Terminal value = final-year FCF × (1 + g) ÷ (WACC − g); perpetual growth defaults to the risk-free rate, kept at least 3 points below WACC
- Equity value = EV − debt + cash + non-operating investments − minority interest, ÷ diluted shares
- Bear / base / bull (growth and margin swings), a WACC × growth sensitivity grid, and an exit-multiple (EV/EBITDA) cross-check
- When consensus shows an acquisition (growth far outside history), the step goes into a pro forma starting revenue rather than the growth rate

## Installation

```bash
# Clone the repository
git clone https://github.com/cadenypatel/mgt6534-dcf-0945-2025.git
cd mgt6534-dcf-0945-2025

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Usage

1. **WACC Calculator**: Enter a ticker symbol and assumptions (risk-free rate, EMRP, credit rating, tax rate) to calculate WACC
2. **Historical Analysis**: Analyze a company's historical financial performance to inform your projections
3. **DCF Model**: Input your WACC and projection assumptions to derive an implied share price

## Key Formulas

| Formula | Description |
|---------|-------------|
| `WACC = w_E × k_E + w_D × k_D × (1-t)` | Weighted Average Cost of Capital |
| `k_E = r_f + β × EMRP` | Cost of Equity (CAPM) |
| `k_D = r_f + spread` | Cost of Debt |
| `UFCF = EBIT(1−t) + D&A − CapEx − ΔNWC` | Unlevered free cash flow |
| `TV = UFCF_n × (1 + g) / (WACC − g)` | Terminal value (Gordon growth) |

All defaults are editable estimates based on reported statements and consensus; they are not guaranteed fair values or investment advice.

## Data Sources

- **Yahoo Finance** (via yfinance): Stock prices, financial statements, market cap, debt, shares outstanding

## Requirements

- Python 3.x
- streamlit
- pandas
- numpy
- yfinance
- statsmodels

## License

MIT License - Free to use for educational purposes

## Author

Caden Patel
