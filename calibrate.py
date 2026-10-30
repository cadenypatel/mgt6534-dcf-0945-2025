"""Calibrate the DCF's default-assumption rules across a broad large-cap sample.

Values every ticker with the app's own default pipeline (no manual inputs)
under each combination of MODEL_RULES, then reports how far model values sit
from market prices. A good default set is unbiased (median value / price near
1) with tight spread; it will not match every ticker.

    python calibrate.py                 # S&P 100, all rule combinations
    python calibrate.py AAPL MSFT TXN   # custom tickers

Tickers in sectors flagged by MODEL_FIT_RULES (banks, insurers, REITs, ...)
are valued but excluded from the summary, since FCFF DCF isn't their method.
"""
import itertools
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

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

RULE_OPTIONS = {
    'sales_to_capital': ['median', 'recency'],
    'margin': ['revert', 'legacy'],
    'growth': ['fy_path', 'ntm'],
    'hold': ['none', 'ceiling'],
}
HORIZON = 10


def gather(ticker):
    """Fetch everything the default valuation needs, once per ticker."""
    risk_free, emrp, tax, rating, _ = app.get_wacc_input_defaults(ticker)
    wacc = app.compute_wacc(ticker, risk_free, emrp, tax, rating, next(iter(app.BETA_METHODS.values())))['wacc']
    company = app.get_company_market_data(ticker)
    ltm_revenue, ltm_date = app.get_ltm_revenue(ticker)
    return {
        'ticker': ticker,
        'risk_free': risk_free,
        'wacc': wacc,
        'company': company,
        'historical': app.get_historical_data(ticker),
        'ltm_revenue': ltm_revenue,
        'ltm_date': ltm_date,
        'forward_growth': app.get_forward_revenue_growth(ticker, ltm_revenue, ltm_date),
        'tax_rate': app.get_effective_tax_rate(ticker),
    }


def as_values(default):
    return list(default) if isinstance(default, (tuple, list)) else [default]


def value_with_rules(inputs, rules):
    """Mirror of the DCF tab's defaults: growth-linked reinvestment, midpoint
    terminal RONIC, risk-free perpetual growth, 10-year horizon."""
    defaults = app.get_projection_defaults_from_data(
        inputs['historical'], inputs['risk_free'], inputs['forward_growth'], inputs['wacc'], rules
    )
    company = inputs['company']
    margins = as_values(defaults['EBIT Margin'])
    case = {
        'ltm_revenue': inputs['ltm_revenue'],
        'ltm_date': inputs['ltm_date'],
        'growth_values': as_values(defaults['Revenue Growth']),
        'horizon': HORIZON,
        'hold_years': app.default_hold_years(inputs['forward_growth'], rules),
        'ebit_margins': np.linspace(margins[0], margins[-1], HORIZON).tolist(),
        'reinv_rates': None,
        'sales_to_capital': defaults['Sales to Capital'],
        'tax_rate': inputs['tax_rate'],
        'total_debt': company['total_debt'],
        'total_cash': company['total_cash'],
        'non_operating_assets': company['non_operating_assets'],
        'minority_interest': company['minority_interest'],
        'shares': company['diluted_shares_outstanding'],
        'fx_rate': company['fx_rate'],
        'da_ratio': defaults['D&A % Revenue'],
    }
    growth = app.default_stable_growth(inputs['risk_free'])
    ronic = app.resolve_terminal_ronic('midpoint', inputs['wacc'], defaults.get('Historical ROC'))
    terminal_reinvestment = app.get_terminal_reinvestment_rate(growth, ronic)
    return app.run_dcf_case(case, inputs['wacc'], growth, terminal_reinvestment)['share_price']


def safe_gather(ticker):
    try:
        return gather(ticker)
    except Exception as error:
        print(f"  skip {ticker}: {str(error)[:90]}", file=sys.stderr)
        return None


def main(tickers):
    with ThreadPoolExecutor(max_workers=3) as pool:
        gathered = [g for g in pool.map(safe_gather, tickers) if g]

    combos = [dict(zip(RULE_OPTIONS, values)) for values in itertools.product(*RULE_OPTIONS.values())]
    rows = []
    for inputs in gathered:
        company = inputs['company']
        excluded = app.get_model_fit_warning(company['sector'], company['industry']) is not None
        for rules in combos:
            try:
                price = value_with_rules(inputs, rules)
            except Exception:
                price = np.nan
            rows.append({'ticker': inputs['ticker'], 'sector': company['sector'], 'excluded': excluded,
                         'market': company['current_price'], 'model': price, **rules})
    results = pd.DataFrame(rows)
    results['log_ratio'] = np.log(results['model'] / results['market'])
    results.to_csv('calibration_results.csv', index=False)

    scored = results[~results['excluded'] & np.isfinite(results['log_ratio'])]
    keys = list(RULE_OPTIONS)
    summary = scored.groupby(keys)['log_ratio'].agg(
        median_bias=lambda s: np.exp(s.median()) - 1,
        median_abs_error=lambda s: np.exp(s.abs().median()) - 1,
        within_25pct=lambda s: (s.abs() <= np.log(1.25)).mean(),
        n='count',
    ).reset_index()
    summary['score'] = summary['median_bias'].abs() + summary['median_abs_error']
    summary = summary.sort_values('score')
    pd.set_option('display.width', 200)
    print(f"\n{scored['ticker'].nunique()} tickers scored ({results['ticker'].nunique()} valued; "
          f"{results.loc[results['excluded'], 'ticker'].nunique()} excluded as poor DCF fits)\n")
    print(summary.to_string(index=False, formatters={
        'median_bias': '{:+.1%}'.format, 'median_abs_error': '{:.1%}'.format,
        'within_25pct': '{:.0%}'.format, 'score': '{:.3f}'.format}))

    best = summary.iloc[0][keys].to_dict()
    chosen = scored[(scored[keys] == pd.Series(best)).all(axis=1)]
    by_sector = chosen.groupby('sector')['log_ratio'].agg(
        median_bias=lambda s: np.exp(s.median()) - 1, n='count').sort_values('median_bias')
    print(f"\nBest rules: {best}\nBias by sector under the best rules:")
    print(by_sector.to_string(formatters={'median_bias': '{:+.1%}'.format}))
    current = scored[(scored[keys] == pd.Series(app.MODEL_RULES)).all(axis=1)]
    print(f"\nCurrent MODEL_RULES {app.MODEL_RULES}: median bias "
          f"{np.exp(current['log_ratio'].median()) - 1:+.1%}, median abs error {np.exp(current['log_ratio'].abs().median()) - 1:.1%}")


if __name__ == "__main__":
    main([t.upper() for t in sys.argv[1:]] or SP100)
