import unittest

import numpy as np
import pandas as pd

import app


def make_case(**overrides):
    case = {
        'revenue': 1000.0, 'start_date': pd.Timestamp("2025-12-31"), 'horizon': 10,
        'growth': 0.10, 'ebit_margin': 0.20, 'tax_rate': 0.25,
        'da_pct': 0.04, 'capex_pct': 0.06, 'nwc_pct': 0.10,
        'total_debt': 200.0, 'total_cash': 50.0, 'non_operating_assets': 20.0, 'minority_interest': 0.0,
        'shares': 10.0, 'fx_rate': 1.0,
    }
    case.update(overrides)
    return case


class ProjectionTests(unittest.TestCase):
    def test_growth_steps_down_to_perpetual_growth(self):
        path = app.growth_path(0.10, 0.04, 10)
        self.assertAlmostEqual(path[0], 0.10)
        self.assertAlmostEqual(path[-1], 0.04)
        np.testing.assert_allclose(np.diff(path), np.diff(path)[0])

    def test_unlevered_fcf_build(self):
        projections = app.build_projections(1000.0, pd.Timestamp("2025-12-31"), [0.10, 0.10], 0.20, 0.25,
                                            0.04, 0.06, 0.10)
        year1 = projections.iloc[0]
        self.assertAlmostEqual(year1['Revenue'], 1100)
        self.assertAlmostEqual(year1['NOPAT'], 1100 * 0.20 * 0.75)
        self.assertAlmostEqual(year1['Change in NWC'], 10)
        self.assertAlmostEqual(year1['Unlevered FCF'], 165 + 44 - 66 - 10)


class ValuationTests(unittest.TestCase):
    def test_gordon_terminal_value_and_bridge(self):
        case = make_case()
        valuation = app.run_dcf(case, 0.09, 0.03)
        final_fcf = valuation['projections']['Unlevered FCF'].iloc[-1]
        self.assertAlmostEqual(valuation['terminal_value'], final_fcf * 1.03 / 0.06)
        discounted = (valuation['projections']['Unlevered FCF'] / 1.09 ** (np.arange(1, 11) - 0.5)).sum()
        self.assertAlmostEqual(valuation['pv_fcf'], discounted)
        self.assertAlmostEqual(valuation['equity_value'], valuation['enterprise_value'] - 200 + 50 + 20)
        self.assertAlmostEqual(valuation['share_price'], valuation['equity_value'] / 10)

    def test_scenarios_are_ordered(self):
        case = make_case()
        bear, base, bull = (app.run_dcf(case, 0.09, 0.03, sign * 0.03, sign * 0.02)['share_price'] for sign in (-1, 0, 1))
        self.assertLess(bear, base)
        self.assertLess(base, bull)

    def test_growth_at_or_above_wacc_rejected(self):
        with self.assertRaises(ValueError):
            app.run_dcf(make_case(), 0.05, 0.05)

    def test_perpetual_growth_default(self):
        self.assertAlmostEqual(app.default_perpetual_growth(0.045, 0.10), 0.045)
        self.assertAlmostEqual(app.default_perpetual_growth(0.05, 0.06), 0.03)


class DefaultsTests(unittest.TestCase):
    def history(self):
        stats = pd.DataFrame({
            "Revenue Growth": [0.10, 0.20, np.nan],
            "EBIT Margin": [0.20, 0.30, 0.90],
            "D&A % Revenue": [0.03, 0.04, 0.05],
            "Capex % Revenue": [0.05, 0.06, 0.07],
            "NWC % Revenue": [0.10, 0.20, 0.90],
        }, index=pd.to_datetime(["2024-12-31", "2025-12-31", "2026-06-30"]))
        return {'df_stats': stats}

    def test_flat_defaults_exclude_overlapping_ltm_from_averages(self):
        defaults = app.get_dcf_defaults(self.history(), 1000.0, 0.12)
        self.assertAlmostEqual(defaults['Revenue Growth'], 0.12)
        self.assertAlmostEqual(defaults['EBIT Margin'], 0.25)
        self.assertAlmostEqual(defaults['NWC % Revenue'], 0.15)
        self.assertAlmostEqual(defaults['Capex % Revenue'], 0.055)
        self.assertAlmostEqual(defaults['D&A % Revenue'], 0.035)

    def test_no_consensus_uses_historical_growth(self):
        defaults = app.get_dcf_defaults(self.history(), 1000.0, None)
        self.assertAlmostEqual(defaults['Revenue Growth'], 0.15)

    def test_acquisition_step_goes_into_starting_revenue(self):
        pending = app.get_dcf_defaults(self.history(), 1000.0, 0.77)
        self.assertAlmostEqual(pending['Revenue'], 1000 * (1 + (0.77 + 0.15) / 2))
        self.assertAlmostEqual(pending['Revenue Growth'], 0.15)
        self.assertTrue(pending['notes'])
        closed = app.get_dcf_defaults(self.history(), 1000.0, 0.77, deal_closed=True)
        self.assertAlmostEqual(closed['Revenue'], 1770)


if __name__ == "__main__":
    unittest.main()
