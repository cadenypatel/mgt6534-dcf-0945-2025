import unittest

import numpy as np
import pandas as pd

import app


def make_case(**overrides):
    case = {
        'ltm_revenue': 1000.0, 'ltm_date': pd.Timestamp("2025-12-31"), 'stub': 0.0, 'horizon': 10,
        'growth_values': [0.10, 0.08, 0.03], 'margin_values': [0.20, 0.22, 0.25], 'capex_values': [0.06],
        'da_pct': 0.04, 'nwc_pct': 0.10, 'tax_rate': 0.21,
        'total_debt': 200.0, 'total_cash': 50.0, 'non_operating_assets': 20.0, 'minority_interest': 0.0,
        'shares': 10.0, 'fx_rate': 1.0,
    }
    case.update(overrides)
    return case


class PathTests(unittest.TestCase):
    def test_explicit_years_then_straight_line_to_final(self):
        path = app.build_path([0.10, 0.08, 0.03], 10)
        self.assertEqual(path[:2], [0.10, 0.08])
        self.assertAlmostEqual(path[-1], 0.03)
        steps = np.diff(path[1:])
        np.testing.assert_allclose(steps, steps[0])

    def test_single_value_is_held(self):
        self.assertEqual(app.build_path([0.05], 4), [0.05] * 4)


class DefaultsTests(unittest.TestCase):
    def stats(self):
        return pd.DataFrame({
            "Revenue Growth": [0.10, 0.20, np.nan],
            "EBIT Margin": [0.20, 0.30, 0.90],
            "D&A % Revenue": [0.03, 0.04, 0.05],
            "Capex % Revenue": [0.05, 0.06, 0.07],
            "NWC % Revenue": [0.10, 0.20, 0.90],
            "Return on Capital": [0.15, 0.25, 0.90],
        }, index=pd.to_datetime(["2024-12-31", "2025-12-31", "2026-06-30"]))

    def test_overlapping_ltm_excluded_from_averages(self):
        defaults = app.get_projection_defaults_from_data({'df_stats': self.stats()}, 0.045)
        self.assertAlmostEqual(defaults['EBIT Margin'][-1], 0.25)
        self.assertAlmostEqual(defaults['NWC % Revenue'], 0.15)
        self.assertAlmostEqual(defaults['Historical ROC'], 0.20)
        # LTM run-rates still seed year 1.
        self.assertAlmostEqual(defaults['EBIT Margin'][0], 0.90)
        self.assertAlmostEqual(defaults['Capex % Revenue'], 0.07)

    def test_return_on_new_capital_excludes_goodwill(self):
        stats = self.stats().assign(**{"ROC ex Goodwill": [0.30, 0.40, 0.90]})
        defaults = app.get_projection_defaults_from_data({'df_stats': stats}, 0.045)
        self.assertAlmostEqual(defaults['Historical ROC'], 0.35)

    def test_unprofitable_history_moves_to_industry_margin(self):
        stats = self.stats().assign(**{"EBIT Margin": [-0.10, -0.05, 0.02]})
        defaults = app.get_projection_defaults_from_data({'df_stats': stats}, 0.045, None, (0.18, "Software"))
        self.assertAlmostEqual(defaults['EBIT Margin'][-1], 0.18)
        profitable = app.get_projection_defaults_from_data({'df_stats': self.stats()}, 0.045, None, (0.18, "Software"))
        self.assertAlmostEqual(profitable['EBIT Margin'][-1], 0.25)

    def test_return_on_new_capital_floored_at_wacc(self):
        self.assertAlmostEqual(app.resolve_terminal_ronic("historical", 0.10, 0.02), 0.10)
        self.assertAlmostEqual(app.resolve_terminal_ronic("historical", 0.10, 0.30), 0.30)
        self.assertAlmostEqual(app.resolve_terminal_ronic("historical", 0.10, None), 0.10)

    def test_perpetual_growth_keeps_spread_to_wacc(self):
        defaults = app.get_projection_defaults_from_data({'df_stats': self.stats()}, 0.05,
                                                         {'growth': (0.10, 0.08), 'margin': None}, None, 0.06)
        self.assertAlmostEqual(defaults['Perpetual Growth'], 0.03)

    def test_acquisition_step_in_consensus_is_averaged_with_history(self):
        deal = {'growth': (0.77, 0.70), 'margin': (0.20, 0.21)}
        defaults = app.get_projection_defaults_from_data({'df_stats': self.stats()}, 0.045, deal)
        self.assertIsNotNone(defaults['_consensus_note'])
        self.assertAlmostEqual(defaults['Revenue Growth'][0], (0.77 + 0.15) / 2)
        self.assertAlmostEqual(defaults['Revenue Growth'][1], 0.15)
        self.assertAlmostEqual(defaults['EBIT Margin'][0], (0.20 + 0.25) / 2)
        closed = app.get_projection_defaults_from_data({'df_stats': self.stats()}, 0.045, deal, deal_closed=True)
        self.assertAlmostEqual(closed['Revenue Growth'][0], 0.77)
        self.assertAlmostEqual(closed['Revenue Growth'][1], 0.15)

    def test_perpetual_growth_capped_at_company_growth(self):
        slow = {'growth': (0.03, 0.028), 'margin': None}
        defaults = app.get_projection_defaults_from_data({'df_stats': self.stats()}, 0.045, slow)
        self.assertAlmostEqual(defaults['Perpetual Growth'], 0.028)
        shrinking = {'growth': (-0.03, -0.02), 'margin': None}
        defaults = app.get_projection_defaults_from_data({'df_stats': self.stats()}, 0.045, shrinking)
        self.assertAlmostEqual(defaults['Perpetual Growth'], app.PERPETUAL_GROWTH_FLOOR)
        fast = {'growth': (0.25, 0.20), 'margin': (0.30, 0.32)}
        defaults = app.get_projection_defaults_from_data({'df_stats': self.stats()}, 0.045, fast)
        self.assertAlmostEqual(defaults['Perpetual Growth'], 0.045)
        self.assertEqual(defaults['EBIT Margin'][:2], (0.30, 0.32))


class EquityModelTests(unittest.TestCase):
    def test_steady_state_matches_justified_price_to_book(self):
        case = {'horizon': 10, 'roe_values': [0.15], 'retention': 0.03 / 0.15, 'book_per_share': 100.0}
        value = app.run_excess_return_case(case, 0.10, 0.03)['share_price']
        self.assertAlmostEqual(value, 100 * (0.15 - 0.03) / (0.10 - 0.03), places=6)

    def test_roe_below_cost_of_equity_gets_no_terminal_growth(self):
        case = {'horizon': 10, 'roe_values': [0.06], 'retention': 0.5, 'book_per_share': 100.0}
        result = app.run_excess_return_case(case, 0.10, 0.05)
        self.assertGreater(result['share_price'], 0)
        self.assertLess(result['share_price'], 100)

    def test_routing(self):
        self.assertTrue(app.uses_equity_model("Financial Services", "Banks - Diversified"))
        self.assertTrue(app.uses_equity_model("Utilities", "Utilities - Regulated Electric"))
        self.assertTrue(app.uses_equity_model("Financial Services", "Credit Services", 0.8))
        self.assertFalse(app.uses_equity_model("Financial Services", "Credit Services", -0.02))
        self.assertFalse(app.uses_equity_model("Technology", "Semiconductors"))


class ValuationTests(unittest.TestCase):
    def test_unlevered_fcf_build(self):
        projections = app.build_dcf_projections(1000.0, pd.Timestamp("2025-12-31"), [0.10, 0.10], [0.20, 0.20],
                                                0.25, 0.04, [0.06, 0.06], 0.10)
        year1 = projections.iloc[0]
        self.assertAlmostEqual(year1['Revenue'], 1100)
        self.assertAlmostEqual(year1['NOPAT'], 1100 * 0.20 * 0.75)
        self.assertAlmostEqual(year1['Change in NWC'], 10)
        self.assertAlmostEqual(year1['Unlevered FCF'], 165 + 44 - 66 - 10)

    def test_final_capex_matches_terminal_reinvestment(self):
        case = make_case()
        ronic, growth = 0.15, 0.03
        final = app.run_dcf_case(case, 0.09, growth, ronic)['projections'].iloc[-1]
        net_investment = final['Capex'] - final['D&A'] + final['Change in NWC']
        self.assertAlmostEqual(net_investment / final['NOPAT'], growth / ronic, places=3)

    def test_scenarios_ordered_and_bridge(self):
        case = make_case()
        bear, base, bull = (app.run_dcf_case(case, 0.09, 0.03, 0.15, sign * 0.03, sign * 0.02)['share_price']
                            for sign in (-1, 0, 1))
        self.assertLess(bear, base)
        self.assertLess(base, bull)
        valuation = app.run_dcf_case(case, 0.09, 0.03, 0.15)
        self.assertAlmostEqual(valuation['equity_value'], valuation['enterprise_value'] - 200 + 50 + 20)

    def test_stub_period_raises_value(self):
        early = app.run_dcf_case(make_case(stub=0.0), 0.09, 0.03, 0.15)['share_price']
        later = app.run_dcf_case(make_case(stub=0.5), 0.09, 0.03, 0.15)['share_price']
        self.assertGreater(later, early)

    def test_growth_at_or_above_wacc_rejected(self):
        with self.assertRaises(ValueError):
            app.run_dcf_case(make_case(), 0.05, 0.05, 0.15)

    def test_reverse_dcf_recovers_base_growth(self):
        case = make_case()
        price = app.run_dcf_case(case, 0.09, 0.03, 0.15)['share_price']
        self.assertAlmostEqual(app.solve_implied_growth(case, 0.09, 0.03, 0.15, price), 0.10, places=4)


if __name__ == "__main__":
    unittest.main()
