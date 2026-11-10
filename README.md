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
- Fetches income statement, balance sheet, and cash flow data from Yahoo Finance
- Computes revenue and EBIT growth rates
- Calculates gross margin and EBIT margin
- Analyzes working capital (NWC and changes)
- Computes reinvestment rate (CapEx - D&A + ΔNWC) / NOPAT

### DCF Model
- Standard banking unlevered free cash flow build from LTM (last twelve months) revenue: UFCF = EBIT × (1 − t) + D&A − capex − ΔNWC
- Revenue growth: analyst consensus for years 1-2, then a straight line to perpetual growth in the final year
- EBIT margin: years 1-2 backed out of consensus GAAP EPS (else the LTM margin), then a straight line to the historical average
- D&A and capex start at their LTM share of revenue; capex moves to the steady-state level the terminal value assumes (capex − D&A + ΔNWC = g ÷ return on new capital × NOPAT); working capital uses its historical average share of revenue
- Perpetual growth defaults to the risk-free rate (long-run nominal growth proxy), capped at the company's consensus year-2 growth and at WACC − 3%
- Companies with no profitable history converge to their industry's operating margin (Damodaran); return on new capital excludes goodwill and is floored at WACC
- Terminal value via Gordon Growth with reinvestment g / RONIC (RONIC defaults to historical average return on capital); mid-year discounting from today, with the stub period since the LTM date
- Cross-checks: exit-multiple terminal value (EV/EBITDA) and a reverse DCF that solves for the market-implied year-1 growth
- Separate bear / base / bull prices (growth and margin swings, not averaged) and a WACC × perpetual-growth sensitivity grid
- Flags sectors where an FCFF DCF is a weak fit (banks, insurers, broker-dealers, REITs, regulated utilities, oil & gas, pre-profit biotech) and names the better method

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
| `TV = NOPAT_n × (1 + g) × (1 - g/RONIC) / (WACC - g)` | Terminal Value (Gordon Growth), with terminal reinvestment implied by `g / RONIC` (return on new invested capital) |

The DCF tab values free cash flow to the firm and is not directly comparable to a dividend discount model. Its long-term growth assumption applies perpetually after the explicit revenue-growth forecast; it is not a forecast-period growth rate. Historical reinvestment × return on capital is shown as a sustainable-growth diagnostic, not automatically treated as perpetual growth: unusually high company growth cannot be extrapolated forever. The recommended stable rate is limited by the risk-free rate (a nominal-growth ceiling), remains editable, and is checked against WACC, consistent with [Damodaran's stable-growth guidance](https://pages.stern.nyu.edu/~adamodar/pdfiles/valonlineslides/session9.pdf). A Gordon Growth terminal value is undefined when long-term growth is at or above WACC.

Every default follows the same rule for every ticker. Revenue growth uses analyst consensus for the two years after the LTM date (fiscal-year consensus blended to the LTM calendar), or the historical average without consensus, then moves in a straight line to the long-term rate in the final forecast year. Margins use consensus GAAP EPS for years 1-2 (non-GAAP EPS is not converted, since it excludes real costs), then move to the historical average margin. Capex starts at the LTM share of revenue and moves to the steady-state level consistent with the terminal value. Long-term growth defaults to the risk-free rate, and the terminal return on new capital defaults to the historical average return on capital (selectable: halfway to WACC, WACC, or custom). Working capital excludes cash and short-term investments. Cash flows use mid-year discounting from the valuation date; equity value adds non-operating investments, subtracts minority interest, and is divided by diluted shares, converted to the quote currency for foreign filers. A latest trailing-twelve-month statement is not treated as an annual growth observation when it overlaps the previous fiscal-year total. Historical return on capital uses average beginning- and ending-period invested capital rather than a single year-end balance-sheet snapshot. These outputs remain estimates based on reported historical statements and user-selected assumptions; they are not guaranteed fair values or investment advice.

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
