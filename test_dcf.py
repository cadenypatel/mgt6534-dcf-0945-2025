import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import app


class DcfCalculationTests(unittest.TestCase):
    def test_historical_cases_use_unweighted_observations(self):
        stats = pd.DataFrame({
            "Revenue Growth": [0.1, 0.3, np.nan],
            "EBIT Margin": [0.2, 0.4, 0.3],
            "Reinv Rate": [0.2, 0.5, 0.8],
        })
        defaults, missing = app.historical_scenario_defaults({"df_stats": stats})
        self.assertEqual(missing, [])
        self.assertAlmostEqual(defaults["Revenue Growth"], 20)
        self.assertAlmostEqual(defaults["EBIT Margin"], 30)
        self.assertAlmostEqual(defaults["Reinv Rate"], 50)

    def test_missing_history_is_reported(self):
        stats = pd.DataFrame({
            "Revenue Growth": [np.nan],
            "EBIT Margin": [0.3],
            "Reinv Rate": [np.inf],
        })
        defaults, missing = app.historical_scenario_defaults({"df_stats": stats})
        self.assertEqual(missing, ["Revenue Growth", "Reinv Rate"])
        self.assertEqual(defaults["Revenue Growth"], 0)

    def test_overlapping_ltm_is_excluded_from_assumption_averages(self):
        stats = pd.DataFrame({
            "Revenue Growth": [0.1, 0.2, np.nan],
            "EBIT Margin": [0.2, 0.4, 0.9],
            "Reinv Rate": [0.2, 0.4, 0.9],
        }, index=pd.to_datetime(["2024-12-31", "2025-12-31", "2026-06-30"]))
        defaults, _ = app.historical_scenario_defaults({"df_stats": stats})
        self.assertAlmostEqual(defaults["EBIT Margin"], 30.0)
        self.assertAlmostEqual(defaults["Reinv Rate"], 30.0)

    def test_symmetric_case_shocks_and_valuation(self):
        cases = app.build_scenario_assumptions(20, 40, 30, 10)
        np.testing.assert_allclose(cases["EBIT margin (%)"], [36, 40, 44])
        np.testing.assert_allclose(cases["Revenue growth (%)"], [20, 20, 20])
        values = {}
        for _, case in cases.iterrows():
            projection = app.build_dcf_projections(
                1000, pd.Timestamp("2025-12-31"), [case["Revenue growth (%)"] / 100] * 5,
                [case["EBIT margin (%)"] / 100] * 5, [case["Reinvestment rate (%)"] / 100] * 5, 0.21
            )
            values[case["Scenario"]] = app.calculate_dcf_valuation(
                projection, 0.09, 0.03, 200, 50, 10, non_operating_assets=20, fx_rate=1.2
            )
        bear, base, best = (values[name]["share_price"] for name in ("Bear", "Base", "Best"))
        self.assertLess(bear, base)
        self.assertLess(base, best)
        self.assertAlmostEqual(base - bear, best - base)
        self.assertIsNone(app.scenario_range_warning(values))

    def test_case_warnings_and_negative_margin(self):
        cases = app.build_scenario_assumptions(0, -20, 25, 10)
        np.testing.assert_allclose(cases["EBIT margin (%)"], [-22, -20, -18])
        for prices, expected in [([120, 100, 130], "not ordered"), ([90, 100, 140], "differ"),
                                 ([90, 100, 110], None)]:
            results = {name: {"share_price": price} for name, price in zip(("Bear", "Base", "Best"), prices)}
            warning = app.scenario_range_warning(results)
            if expected is None:
                self.assertIsNone(warning)
            else:
                self.assertIn(expected, warning)
        with self.assertRaises(ValueError):
            app.build_scenario_assumptions(0, 0, 0, -1)

    def test_outlying_history_does_not_make_case_offsets_asymmetric(self):
        stats = pd.DataFrame({
            "Revenue Growth": [0.05, 0.06, 0.50],
            "EBIT Margin": [0.10, 0.12, 0.80],
            "Reinv Rate": [0.20, 0.25, 0.30],
        })
        defaults, _ = app.historical_scenario_defaults({"df_stats": stats})
        cases = app.build_scenario_assumptions(
            defaults["Revenue Growth"], defaults["EBIT Margin"], defaults["Reinv Rate"], 10
        )
        values = []
        for _, case in cases.iterrows():
            projection = app.build_dcf_projections(
                1000, pd.Timestamp("2025-12-31"), [case["Revenue growth (%)"] / 100] * 5,
                [case["EBIT margin (%)"] / 100] * 5, [case["Reinvestment rate (%)"] / 100] * 5, 0.21
            )
            values.append(app.calculate_dcf_valuation(projection, 0.09, 0.03, 0, 0, 10)["share_price"])
        self.assertAlmostEqual(values[1] - values[0], values[2] - values[1])

    def test_quarterly_ltm_is_not_restricted_to_2026_or_partial_sums(self):
        dates = pd.to_datetime(["2026-06-30", "2026-09-30", "2026-12-31", "2027-03-31"])
        income = pd.DataFrame({
            date: {"Total Revenue": 100.0, "Operating Income": 20.0,
                   "Gross Profit": (np.nan if date == dates[0] else 40.0)}
            for date in dates
        })
        ticker = SimpleNamespace(quarterly_income_stmt=income, quarterly_financials=income,
                                 quarterly_balance_sheet=pd.DataFrame(),
                                 quarterly_cashflow=pd.DataFrame())
        annual = pd.DataFrame({"Total Revenue": [350.0]}, index=pd.to_datetime(["2026-12-31"]))
        with patch.object(app.yf, "Ticker", return_value=ticker):
            augmented, _, _ = app._augment_with_latest_quarterly(
                "TEST", annual, pd.DataFrame(), pd.DataFrame()
            )
        self.assertEqual(augmented.loc[dates[-1], "Total Revenue"], 400)
        self.assertEqual(augmented.loc[dates[-1], "EBIT"], 80)
        self.assertTrue(pd.isna(augmented.loc[dates[-1], "Gross Profit"]))

    def test_historical_reinvestment_does_not_replace_missing_inputs_with_zero(self):
        dates = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"])
        income = pd.DataFrame({
            "Total Revenue": [100, 110, 120, 130], "Operating Income": [20, 22, 24, 26],
            "Gross Profit": [50, 55, 60, 65], "Pretax Income": [18, 20, 22, 24],
            "Tax Provision": [4, 4, 5, 5],
        }, index=dates)
        balance = pd.DataFrame({
            "Current Liabilities": [20, 21, 22, 23], "Cash And Cash Equivalents": [5, 5, 5, 5],
            "Current Debt": [1, 1, 1, 1], "Total Debt": [10, 10, 10, 10],
            "Stockholders Equity": [50, 55, 60, 65],
        }, index=dates)
        cashflow = pd.DataFrame({
            "Capital Expenditure": [-10, -10, -10, -10],
            "Depreciation And Amortization": [5, 5, 5, 5],
        }, index=dates)
        ticker = SimpleNamespace(quarterly_income_stmt=pd.DataFrame(),
                                 quarterly_financials=pd.DataFrame())
        with patch.object(app, "get_sec_historical_statements", return_value=(income, balance, cashflow)), \
                patch.object(app.yf, "Ticker", return_value=ticker):
            data = app.get_historical_data("TEST")
        self.assertTrue(data["df_stats"]["Reinv Rate"].isna().all())
        _, missing = app.historical_scenario_defaults(data)
        self.assertIn("Reinv Rate", missing)

    def test_mid_year_valuation_and_equity_bridge(self):
        projections = app.build_dcf_projections(
            100, pd.Timestamp("2025-12-31"), [0.1, 0.1], [0.2, 0.2], [0.25, 0.25], 0.2
        )
        self.assertAlmostEqual(projections.iloc[0]["FCF"], 13.2)
        self.assertAlmostEqual(projections.iloc[1]["FCF"], 14.52)
        result = app.calculate_dcf_valuation(
            projections, 0.1, 0.03, 20, 10, 5,
            non_operating_assets=4, minority_interest=2, fx_rate=2
        )
        pv_fcf = 13.2 / 1.1 ** 0.5 + 14.52 / 1.1 ** 1.5
        terminal_value = 14.52 * 1.03 / 0.07
        enterprise = pv_fcf + terminal_value / 1.1 ** 1.5
        self.assertAlmostEqual(result["pv_fcf"], pv_fcf)
        self.assertAlmostEqual(result["terminal_value"], terminal_value)
        self.assertAlmostEqual(result["enterprise_value"], enterprise)
        self.assertAlmostEqual(result["share_price"], (enterprise - 20 + 10 + 4 - 2) * 2 / 5)
        self.assertNotIn("Discounted_FCF", projections.columns)

    def test_invalid_assumptions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "projection year"):
            app.build_dcf_projections(100, pd.Timestamp("2025-12-31"), [0.1], [], [0.2], 0.2)
        with self.assertRaisesRegex(ValueError, "below -100%"):
            app.build_dcf_projections(100, pd.Timestamp("2025-12-31"), [-1.01], [0.2], [0.2], 0.2)
        projections = app.build_dcf_projections(100, pd.Timestamp("2025-12-31"), [0.1], [0.2], [0.2], 0.2)
        for wacc, growth, shares, fx in [(0.1, 0.1, 1, 1), (0.1, -1, 1, 1),
                                          (0.1, 0.02, 0, 1), (0.1, 0.02, 1, 0)]:
            with self.subTest(wacc=wacc, growth=growth, shares=shares, fx=fx):
                with self.assertRaises(ValueError):
                    app.calculate_dcf_valuation(projections, wacc, growth, 0, 0, shares, fx_rate=fx)


UI_SCRIPT = """
import pandas as pd
import app

company = {
    'company_name': 'Example', 'market_cap': 1000000000, 'total_debt': 200000000,
    'total_cash': 50000000, 'non_operating_assets': 10000000, 'minority_interest': 5000000,
    'diluted_shares_outstanding': 10000000, 'current_price': 20,
    'financial_currency': 'USD', 'price_currency': 'USD', 'fx_rate': 1.0,
}
app.get_wacc_input_defaults = lambda ticker: (0.04, 0.05, 0.25, 'Baa2/BBB', 4.0)
app.get_credit_spread = lambda rating: 0.01
app.get_company_market_data = lambda ticker: company
app.calculate_beta = lambda ticker: (1.3, 0.7, None, None)
app.get_bottom_up_beta = lambda ticker, debt_to_equity, tax: (1.2, 'Industry', 1.0)
app.get_effective_tax_rate = lambda ticker: 0.2
app.get_ltm_revenue = lambda ticker: (1000000000, pd.Timestamp('2025-12-31'))
app.get_historical_data = lambda ticker: {'df_stats': pd.DataFrame({
    'Revenue Growth': [0.05, 0.10, 0.15],
    'EBIT Margin': [0.20, 0.25, 0.30],
    'Reinv Rate': [0.20, 0.25, 0.30],
})}
app.render_wacc()
app.render_dcf()
"""


class DcfWorkflowTests(unittest.TestCase):
    def test_editable_wacc_and_scenario_flow(self):
        ui = AppTest.from_string(UI_SCRIPT, default_timeout=20).run()
        self.assertFalse(ui.exception)
        self.assertFalse(ui.error)
        self.assertAlmostEqual(ui.number_input(key="wacc_beta_input").value, 1.2)
        base_before = next(metric.value for metric in ui.metric if metric.label == "Base")
        ui.number_input(key="wacc_beta_input").set_value(1.5)
        ui.number_input(key="wacc_spread_input").set_value(2.0)
        ui.button[0].click().run()
        self.assertFalse(ui.exception)
        expected_wacc = (1 / 1.2 * (0.04 + 1.5 * 0.05)
                         + 0.2 / 1.2 * (0.04 + 0.02) * 0.75)
        self.assertAlmostEqual(ui.number_input(key="dcf_wacc_rate").value, expected_wacc * 100)
        base_after_wacc = next(metric.value for metric in ui.metric if metric.label == "Base")
        self.assertNotEqual(base_before, base_after_wacc)

        ui.number_input(key="dcf_base_growth").set_value(30.0).run()
        self.assertFalse(ui.exception)
        base_after_scenario = next(metric.value for metric in ui.metric if metric.label == "Base")
        self.assertNotEqual(base_after_wacc, base_after_scenario)
        ui.checkbox(key="dcf_custom_scenarios").check().run()
        scenario_key = f"dcf_scenarios_{ui.session_state['dcf_scenario_revision']}"
        ui.session_state[scenario_key] = {
            "edited_rows": {2: {"Revenue growth (%)": 80.0}},
            "added_rows": [], "deleted_rows": [],
        }
        ui.run()
        self.assertTrue(any("ranges differ" in warning.value for warning in ui.warning))
        ui.number_input(key="dcf_cash").set_value(70.0).run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.number_input(key="dcf_cash").value, 70.0)
        self.assertNotEqual(base_after_scenario, next(metric.value for metric in ui.metric if metric.label == "Base"))

    def test_symmetric_defaults_update_with_base_and_shock(self):
        ui = AppTest.from_string(UI_SCRIPT, default_timeout=20).run()
        def prices():
            return [float(next(metric.value for metric in ui.metric if metric.label == label).split()[0].replace(",", ""))
                    for label in ("Bear", "Base", "Best")]
        bear, base, best = prices()
        self.assertAlmostEqual(base - bear, best - base, delta=0.02)
        ui.number_input(key="dcf_base_margin").set_value(40.0)
        ui.number_input(key="dcf_margin_change").set_value(20.0).run()
        bear, base, best = prices()
        self.assertLess(bear, base)
        self.assertLess(base, best)
        self.assertAlmostEqual(base - bear, best - base, delta=0.02)
        ui.number_input(key="dcf_base_margin").set_value(0.0).run()
        self.assertTrue(any("identical cases" in warning.value for warning in ui.warning))

    def test_invalid_terminal_and_ticker_refresh(self):
        ui = AppTest.from_string(UI_SCRIPT, default_timeout=20).run()
        ui.number_input(key="dcf_terminal_growth").set_value(10.0).run()
        self.assertTrue(any("lower than WACC" in error.value for error in ui.error))
        ui.text_input(key="shared_ticker_symbol").set_value("OTHER").run()
        self.assertFalse(ui.exception)
        self.assertFalse(ui.error)
        self.assertEqual(ui.number_input(key="dcf_terminal_growth").value, 3.0)
        self.assertEqual(ui.session_state["dcf_loaded_ticker"], "OTHER")

    def test_missing_data_warns_and_allows_manual_revenue(self):
        script = UI_SCRIPT.replace(
            "app.render_wacc()",
            "def fail_revenue(ticker):\n    raise ValueError('unavailable')\n"
            "app.get_ltm_revenue = fail_revenue\napp.render_wacc()",
        )
        ui = AppTest.from_string(script, default_timeout=20).run()
        self.assertFalse(ui.exception)
        self.assertTrue(any("Could not load LTM revenue" in warning.value for warning in ui.warning))
        self.assertTrue(any("LTM revenue must be greater than zero" in error.value for error in ui.error))
        ui.number_input(key="dcf_revenue").set_value(1000.0).run()
        self.assertFalse(ui.error)
        self.assertTrue(any(metric.label == "Base" for metric in ui.metric))


if __name__ == "__main__":
    unittest.main()
