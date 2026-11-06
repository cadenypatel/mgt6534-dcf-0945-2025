# DCF Equity Valuation App

A Streamlit web application for Discounted Cash Flow (DCF) equity valuation. Value any publicly traded company by computing WACC, analyzing historical financials, and building a complete DCF model.



## Live Demo

Deploy your own instance via [Streamlit Cloud](https://share.streamlit.io)

## Features

### WACC Calculator
- Beta (default): average of the Blume-adjusted OLS regression beta (5-year monthly returns vs S&P 500) and Damodaran's bottom-up industry beta relevered at the firm's D/E; either can be selected alone
- Equity risk premium, industry betas, and rating spreads are fetched live from Damodaran's latest datasets (cached daily, with stored fallbacks)
- Cost of equity using CAPM (risk-free rate + beta × equity market risk premium)
- Cost of debt using risk-free rate + credit spread (Damodaran lookup table)
- Beta, risk-free rate, equity risk premium, credit spread, marginal tax rate, market capitalization, and debt are editable
- Weighted average based on market capitalization and total debt; apply the calculated WACC to the DCF or enter one directly

### Historical Analysis
- Fetches income statement, balance sheet, and cash flow data from Yahoo Finance
- Computes revenue and EBIT growth rates
- Calculates gross margin and EBIT margin
- Analyzes working capital (NWC and changes)
- Computes reinvestment rate (CapEx - D&A + ΔNWC) / NOPAT

### DCF Model
- Uses LTM (Last Twelve Months) revenue as starting point
- Bear, base, and best cases with editable revenue growth, EBIT margin, and reinvestment rates, held constant throughout each forecast
- Base assumptions use arithmetic means of available annual historical data; overlapping LTM observations are excluded
- Default bear and best cases vary growth, margin, and reinvestment together around the base. The starting changes (±2, ±1, and ±2 percentage points, respectively) and individual case assumptions are editable; WACC and terminal growth are shared
- Editable forecast horizon, LTM revenue and date, tax rate, WACC, perpetual growth, debt, cash, non-operating investments, minority interest, diluted shares, exchange rate, and market price
- Gordon Growth terminal value and mid-year discounted cash flows; displays a forecast and equity-value bridge for each selected case
- Perpetual growth must be below WACC; review the continuing reinvestment assumption for consistency with perpetual growth

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
3. **DCF Model**: Review the fetched inputs, edit the three scenario assumptions, and compare the implied share prices and valuation bridges

## Key Formulas

| Formula | Description |
|---------|-------------|
| `WACC = w_E × k_E + w_D × k_D × (1-t)` | Weighted Average Cost of Capital |
| `k_E = r_f + β × EMRP` | Cost of Equity (CAPM) |
| `k_D = r_f + spread` | Cost of Debt |
| `FCF = NOPAT × (1 - Reinvestment Rate)` | Free Cash Flow |
| `TV = FCF_n × (1 + g) / (WACC - g)` | Terminal Value (Gordon Growth) |

The model values free cash flow to the firm. Perpetual growth is separate from the forecast-period assumptions and must be lower than WACC. Terminal FCF grows the final forecast year's FCF, so its reinvestment rate is implicitly continued in perpetuity. Users should ensure that rate is sufficient to support their chosen perpetual growth, particularly for capital-intensive businesses.

Scenario analysis combines assumptions into coherent business narratives. The illustrative starting cases assume slower growth, lower margins, and less reinvestment in the bear case, and the reverse in the best case. Those changes are placeholders, not analyst forecasts or prescribed industry standards; depending on the business, weaker growth might require *more* reinvestment, or stronger growth might require less. Edit each case accordingly and check whether reinvestment is sufficient to support growth. WACC, perpetual growth, tax, and the equity bridge are shared across cases so the operating assumptions drive the comparison.

The output range is not forced to be symmetric: compounding and the equity bridge can make upside and downside differ even when assumptions change by equal amounts. The app warns if prices are out of order. See [CFI's scenario analysis guide](https://corporatefinanceinstitute.com/resources/financial-modeling/scenario-analysis/) and [Oxxon Advisors' DCF scenario analysis guide](https://www.oxxonadvisors.com/fmae-knowledge-centre/technical-guides/dcf-scenario-analysis/) for discussion of scenario construction.

When data is unavailable, the app identifies placeholders for the user to replace. Financial data and share counts can be incomplete or overlap (for example cash and investments), so verify the equity bridge and FX conversion. Results are estimates, not guaranteed fair values or investment advice.

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
