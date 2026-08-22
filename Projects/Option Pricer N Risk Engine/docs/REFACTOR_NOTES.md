
# Refactor notes

The original 6,380-line script is preserved verbatim in
`archive/original_monolith.py`. The notebook replaces only the monolithic
`main()` orchestration. All original top-level classes and functions are kept in
the package, with the following deliberate corrections:

- Black-Scholes resolves default volatility before calculating d1/d2.
- Black-Scholes handles `time_to_expiry == 0` using intrinsic value.
- Logging is configured once by the notebook rather than at import time.
- Historical download and feature engineering are named functions so they can
  be tested and persisted.
- Database and validation layers were added without changing the financial
  logic.

| Original definition | Original lines | Refactored module |
|---|---:|---|
| `OptionTicker` | 109-311 | `src/options_risk_engine/domain.py` |
| `make_model_specs` | 314-359 | `src/options_risk_engine/forecasting/volatility.py` |
| `black_scholes` | 363-433 | `src/options_risk_engine/pricing/black_scholes.py` |
| `create_model` | 438-502 | `src/options_risk_engine/forecasting/volatility.py` |
| `calculate_metrics` | 506-637 | `src/options_risk_engine/forecasting/volatility.py` |
| `garch_predictions_for_block` | 640-831 | `src/options_risk_engine/forecasting/volatility.py` |
| `evaluate_symbol_models` | 834-1356 | `src/options_risk_engine/forecasting/volatility.py` |
| `evaluate_historical_mean_holdout` | 1359-1428 | `src/options_risk_engine/forecasting/volatility.py` |
| `monte_carlo_option_chain` | 1432-1563 | `src/options_risk_engine/pricing/monte_carlo.py` |
| `years_to_expiry` | 1595-1609 | `src/options_risk_engine/pricing/black_scholes.py` |
| `latest_feature_value` | 1613-1639 | `src/options_risk_engine/data/market_data.py` |
| `safe_relative_edge` | 1643-1665 | `src/options_risk_engine/utils.py` |
| `choose_closest_expiry` | 1673-1690 | `src/options_risk_engine/pricing/chains.py` |
| `clean_downloaded_chain` | 1693-1720 | `src/options_risk_engine/pricing/chains.py` |
| `download_option_chains` | 1724-1790 | `src/options_risk_engine/pricing/chains.py` |
| `get_volatility_inputs` | 1800-1836 | `src/options_risk_engine/pricing/chains.py` |
| `as_aligned_series` | 1840-1854 | `src/options_risk_engine/pricing/chains.py` |
| `calculate_black_scholes_scenarios` | 1858-1937 | `src/options_risk_engine/pricing/chains.py` |
| `build_comparison_table` | 1942-1996 | `src/options_risk_engine/pricing/chains.py` |
| `check_put_call_parity` | 2006-2092 | `src/options_risk_engine/pricing/chains.py` |
| `attach_parity_errors` | 2096-2121 | `src/options_risk_engine/pricing/chains.py` |
| `add_recommendation_metrics` | 2130-2177 | `src/options_risk_engine/pricing/chains.py` |
| `filter_highlighted_options` | 2182-2208 | `src/options_risk_engine/pricing/chains.py` |
| `option_price_bounds` | 2210-2251 | `src/options_risk_engine/pricing/black_scholes.py` |
| `price_option_universe` | 2254-2649 | `src/options_risk_engine/pricing/chains.py` |
| `print_highlighted_options` | 2653-2682 | `src/options_risk_engine/reporting/tables.py` |
| `plot_monte_carlo_result` | 2691-2728 | `src/options_risk_engine/pricing/monte_carlo.py` |
| `add_option_profit_distribution` | 2732-2811 | `src/options_risk_engine/pricing/monte_carlo.py` |
| `run_monte_carlo_analysis` | 2815-2924 | `src/options_risk_engine/pricing/monte_carlo.py` |
| `combine_option_tables` | 2933-2963 | `src/options_risk_engine/reporting/tables.py` |
| `create_findings_table` | 2966-3024 | `src/options_risk_engine/reporting/tables.py` |
| `print_final_tables` | 3029-3122 | `src/options_risk_engine/reporting/tables.py` |
| `calendar_days_to_trading_days` | 3126-3155 | `src/options_risk_engine/pricing/greeks.py` |
| `objective_function` | 3157-3178 | `src/options_risk_engine/pricing/black_scholes.py` |
| `black_scholes_greeks` | 3181-3403 | `src/options_risk_engine/pricing/greeks.py` |
| `add_greek_columns` | 3406-3493 | `src/options_risk_engine/pricing/greeks.py` |
| `add_greek_sign_checks` | 3496-3614 | `src/options_risk_engine/pricing/greeks.py` |
| `scalar_black_scholes_price` | 3617-3652 | `src/options_risk_engine/pricing/black_scholes.py` |
| `numerical_black_scholes_greeks` | 3654-3842 | `src/options_risk_engine/pricing/greeks.py` |
| `add_numerical_greek_validation` | 3844-3986 | `src/options_risk_engine/pricing/greeks.py` |
| `implied_volatility` | 3989-4072 | `src/options_risk_engine/pricing/implied_volatility.py` |
| `add_implied_volatility_columns` | 4075-4138 | `src/options_risk_engine/pricing/implied_volatility.py` |
| `risk_engine_data_prep` | 4141-4378 | `src/options_risk_engine/risk/portfolio.py` |
| `risk_engine_summary` | 4380-4487 | `src/options_risk_engine/risk/portfolio.py` |
| `generate_random_scenarios` | 4490-4599 | `src/options_risk_engine/risk/scenarios.py` |
| `run_scenario_engine` | 4602-4981 | `src/options_risk_engine/risk/scenarios.py` |
| `analyse_scenario_results` | 4984-5422 | `src/options_risk_engine/risk/scenarios.py` |
| `calculate_greek_profit_loss` | 5424-5628 | `src/options_risk_engine/risk/attribution.py` |
| `analyse_greek_attribution` | 5631-6052 | `src/options_risk_engine/risk/attribution.py` |
