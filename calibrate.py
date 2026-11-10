"""Check the DCF's default assumptions across a broad large-cap sample.

Values every ticker with the app's own default pipeline (no manual inputs)
and reports how far model values sit from market prices. Good defaults are
unbiased (median value / price near 1); no default set will match every
ticker, because the market prices in views the defaults don't hold.

    python calibrate.py                 # S&P 100
    python calibrate.py AAPL MSFT TXN   # custom tickers

Banks, insurers, lenders and regulated utilities are valued with the equity
(excess-return) model, as in the app. Remaining poor DCF fits (REITs, oil &
gas, pre-profit biotech) are valued but excluded from the summary.
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

import app

SP100 = """AAPL ABBV ABT ACN ADBE AIG AMD AMGN AMT AMZN AVGO AXP BA BAC BK BKNG BLK BMY BRK-B C CAT CHTR CL
CMCSA COF COP COST CRM CSCO CVS CVX DE DHR DIS DUK EMR F FDX GD GE GILD GM GOOGL GS HD HON IBM INTC INTU
ISRG JNJ JPM KHC KO LIN LLY LMT LOW MA MCD MDLZ MDT MET META MMM MO MRK MS MSFT NEE NFLX NKE NVDA ORCL
PEP PFE PG PLTR PM PYPL QCOM RTX SBUX SCHW SO SPG T TGT TMO TMUS TSLA TXN UNH UNP UPS USB V VZ WFC WMT XOM""".split()
HORIZON = 10


def value(ticker):
    """Default valuation for one ticker, mirroring the WACC and DCF tabs."""
    risk_free, emrp, marginal_tax, rating, _ = app.get_wacc_input_defaults(ticker)
    capital = app.compute_wacc(ticker, risk_free, emrp, marginal_tax, rating, "blend")
    wacc = capital['wacc']
    company = app.get_company_market_data(ticker)
    interest_share = (app.get_financial_company_data(ticker)['net_interest_share']
                      if company['sector'] == "Financial Services" else None)
    if app.uses_equity_model(company['sector'], company['industry'], interest_share):
        data = app.get_financial_company_data(ticker)
        equity = app.get_financial_defaults(data, company['diluted_shares_outstanding'], company['fx_rate'],
                                            capital['cost_of_equity'], risk_free)
        case = {'horizon': HORIZON, 'roe_values': list(equity['roe_values']), 'retention': equity['retention'],
                'book_per_share': equity['book_per_share']}
        price = app.run_excess_return_case(case, capital['cost_of_equity'], equity['perpetual_growth'])['share_price']
        return {'ticker': ticker, 'sector': company['sector'], 'method': 'equity', 'excluded': False,
                'market': company['current_price'], 'model': price, 'exit_multiple': np.nan,
                'wacc': capital['cost_of_equity'], 'consensus': data['consensus_eps'] is not None}
    historical = app.get_historical_data(ticker)
    tax_rate = app.get_effective_tax_rate(ticker)
    ltm_revenue, ltm_date = app.get_ltm_revenue(ticker)
    consensus = app.get_consensus_forecast(ticker, ltm_revenue, ltm_date, historical, tax_rate,
                                           company['diluted_shares_outstanding'], company['fx_rate'])
    defaults = app.get_projection_defaults_from_data(historical, risk_free, consensus,
                                                     app.get_industry_operating_margin(ticker), wacc)
    growth = defaults['Perpetual Growth']
    ronic = app.resolve_terminal_ronic('historical', wacc, defaults['Historical ROC'])
    case = {
        'ltm_revenue': ltm_revenue, 'ltm_date': ltm_date, 'stub': app.stub_years(ltm_date), 'horizon': HORIZON,
        'growth_values': list(defaults['Revenue Growth']), 'margin_values': list(defaults['EBIT Margin']),
        'capex_values': [defaults['Capex % Revenue']], 'da_pct': defaults['D&A % Revenue'],
        'nwc_pct': defaults['NWC % Revenue'], 'tax_rate': tax_rate,
        'total_debt': company['total_debt'], 'total_cash': company['total_cash'],
        'non_operating_assets': company['non_operating_assets'], 'minority_interest': company['minority_interest'],
        'shares': company['diluted_shares_outstanding'], 'fx_rate': company['fx_rate'],
    }
    valuation = app.run_dcf_case(case, wacc, growth, ronic)
    multiple = app.current_ev_to_ebitda(company, historical)
    exit_price = app.exit_multiple_price(valuation, case, wacc, multiple)[0] if multiple else np.nan
    return {
        'ticker': ticker, 'sector': company['sector'], 'method': 'dcf',
        'excluded': app.get_model_fit_warning(company['sector'], company['industry']) is not None,
        'market': company['current_price'], 'model': valuation['share_price'], 'exit_multiple': exit_price,
        'wacc': wacc, 'consensus': consensus is not None,
    }


def safe_value(ticker):
    try:
        return value(ticker)
    except Exception as error:
        print(f"  skip {ticker}: {str(error)[:90]}", file=sys.stderr)
        return None


def summarize(log_ratio):
    return pd.Series({
        'median_bias': np.exp(log_ratio.median()) - 1,
        'median_abs_error': np.exp(log_ratio.abs().median()) - 1,
        'within_25pct': (log_ratio.abs() <= np.log(1.25)).mean(),
        'n': log_ratio.count(),
    })


def main(tickers):
    # One ticker at a time: yfinance mixes data between tickers fetched in parallel.
    results = pd.DataFrame([row for row in map(safe_value, tickers) if row])
    results['log_ratio'] = np.log(results['model'] / results['market'])
    results['exit_log_ratio'] = np.log(results['exit_multiple'] / results['market'])
    results.to_csv('calibration_results.csv', index=False)

    scored = results[~results['excluded'] & np.isfinite(results['log_ratio'])]
    formats = {'median_bias': '{:+.1%}'.format, 'median_abs_error': '{:.1%}'.format,
               'within_25pct': '{:.0%}'.format, 'n': '{:.0f}'.format}
    print(f"\n{len(scored)} tickers scored ({len(results)} valued; "
          f"{int(results['excluded'].sum())} excluded as poor DCF fits)\n")
    summary = pd.DataFrame({
        'Perpetual growth (base)': summarize(scored['log_ratio']),
        'Exit multiple (cross-check)': summarize(scored['exit_log_ratio'].replace([np.inf, -np.inf], np.nan)),
    }).T
    print(summary.to_string(formatters=formats))
    by_sector = scored.groupby('sector')['log_ratio'].apply(summarize).unstack()
    print("\nBias by sector (perpetual growth):")
    print(by_sector.sort_values('median_bias').to_string(formatters=formats))
    by_method = scored.groupby('method')['log_ratio'].apply(summarize).unstack()
    print("\nBy method:")
    print(by_method.to_string(formatters=formats))


if __name__ == "__main__":
    main([t.upper() for t in sys.argv[1:]] or SP100)
