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
  - Two assumptions transition linearly from year 1 to the final forecast year
- Terminal value via Gordon Growth Model
- Discounts all cash flows to present value
- Computes implied share price vs current market price
- Sensitivity analysis for different WACC scenarios
- Uncapped, editable annual and long-term growth assumptions; the recommended perpetual rate is bounded by the risk-free rate as a long-run nominal-growth ceiling, and must remain below WACC for a finite terminal value

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
| `FCF = NOPAT - ΔRevenue / (Sales-to-Capital)` | Free Cash Flow; reinvestment is derived from growth (a manual reinvestment rate is optional) |
| `TV = NOPAT_n × (1 + g) × (1 - g/RONIC) / (WACC - g)` | Terminal Value (Gordon Growth), with terminal reinvestment implied by `g / RONIC` (return on new invested capital) |

The DCF tab values free cash flow to the firm and is not directly comparable to a dividend discount model. Its long-term growth assumption applies perpetually after the explicit revenue-growth forecast; it is not a forecast-period growth rate. Historical reinvestment × return on capital is shown as a sustainable-growth diagnostic, not automatically treated as perpetual growth: unusually high company growth cannot be extrapolated forever. The recommended stable rate is limited by the risk-free rate (a nominal-growth ceiling), remains editable, and is checked against WACC, consistent with [Damodaran's stable-growth guidance](https://pages.stern.nyu.edu/~adamodar/pdfiles/valonlineslides/session9.pdf). A Gordon Growth terminal value is undefined when long-term growth is at or above WACC.

Revenue growth starts at analyst-consensus next-twelve-months growth (or the latest reported year) and converges to the long-term rate by the final forecast year, closing about 25% of the gap each year to reflect empirical mean reversion of growth. Margins transition linearly between two values. Reinvestment is derived from growth through the historical sales-to-capital ratio. Long-term growth defaults to the risk-free rate, and the terminal return on new capital defaults to halfway between historical return on capital and WACC (selectable: WACC, historical, or custom). Cash flows use mid-year discounting; equity value adds non-operating investments, subtracts minority interest, and is divided by diluted shares, converted to the quote currency for foreign filers. A latest trailing-twelve-month statement is not treated as an annual growth observation when it overlaps the previous fiscal-year total. Historical return on capital uses average beginning- and ending-period invested capital rather than a single year-end balance-sheet snapshot. These outputs remain estimates based on reported historical statements and user-selected assumptions; they are not guaranteed fair values or investment advice.

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
