# DCF Equity Valuation App

A Streamlit web application for Discounted Cash Flow (DCF) equity valuation. Value any publicly traded company by computing WACC, analyzing historical financials, and building a complete DCF model.



## Live Demo

Deploy your own instance via [Streamlit Cloud](https://share.streamlit.io)

## Features

### WACC Calculator
- Calculates beta via OLS regression (5-year monthly returns vs S&P 500)
- Cost of equity using CAPM (risk-free rate + beta × equity market risk premium)
- Cost of debt using risk-free rate + credit spread (Damodaran lookup table)
- Weighted average based on market cap and total debt

### Historical Analysis
- Fetches income statement, balance sheet, and cash flow data from Yahoo Finance
- Computes revenue and EBIT growth rates
- Calculates gross margin and EBIT margin
- Analyzes working capital (NWC and changes)
- Computes reinvestment rate (CapEx - D&A + ΔNWC) / NOPAT

### DCF Model
- Uses LTM (Last Twelve Months) revenue as starting point
- Customizable projection assumptions:
  - Revenue growth rates by year
  - EBIT margins by year
  - Reinvestment rates by year
- Terminal value via Gordon Growth Model
- Discounts all cash flows to present value
- Computes implied share price vs current market price
- Sensitivity analysis for different WACC scenarios

## Installation

```bash
# Clone the repository
git clone https://github.com/jblocher/mgt6534-dcf-0945-2025.git
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
| `FCF = NOPAT × (1 - Reinvestment Rate)` | Free Cash Flow |
| `TV = FCF × (1 + g) / (WACC - g)` | Terminal Value (Gordon Growth) |

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

Caden Patel - Vanderbilt University
