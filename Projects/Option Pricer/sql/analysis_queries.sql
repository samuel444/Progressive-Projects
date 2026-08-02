
-- Latest run
SELECT * FROM project_runs ORDER BY created_at DESC LIMIT 1;

-- Worst portfolio scenarios for a run
SELECT "Scenario ID", Portfolio_PnL, Portfolio_Value
FROM scenario_portfolio_results
WHERE run_id = :run_id AND "Scenario ID" <> 'BASE'
ORDER BY Portfolio_PnL ASC
LIMIT 10;

-- Largest ticker losses in one scenario
SELECT Ticker, Scenario_PnL, Scenario_Value
FROM scenario_ticker_results
WHERE run_id = :run_id AND "Scenario ID" = :scenario_id
ORDER BY Scenario_PnL ASC;

-- Attribution scenarios with the largest gross residual percentage
SELECT *
FROM scenario_attribution
WHERE run_id = :run_id AND "Scenario ID" <> 'BASE'
ORDER BY "Gross Residual %" DESC
LIMIT 10;
