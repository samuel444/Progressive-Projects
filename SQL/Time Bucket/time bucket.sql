-- By Year
SELECT
    strftime('%Y', trade_time) AS year,
    MAX(share_price_change) AS max_change
FROM SharePrices
GROUP BY strftime('%Y', trade_time)
ORDER BY year;


-- By Month
SELECT
    strftime('%Y-%m', trade_time) AS month,
    MIN(share_price_change) AS min_change
FROM SharePrices
GROUP BY strftime('%Y-%m', trade_time)
ORDER BY month;


-- By Day
SELECT
    DATE(trade_time) AS day,
    AVG(share_price_change) AS average_change
FROM SharePrices
GROUP BY DATE(trade_time)
ORDER BY day;


-- By Hour of the Day (combines all days)
SELECT
    strftime('%H', trade_time) AS hour,
    SUM(share_price_change) AS overall_return
FROM SharePrices
GROUP BY strftime('%H', trade_time)
ORDER BY hour;


-- By Individual Hour
SELECT
    strftime('%Y-%m-%d %H', trade_time) AS hourly_bucket,
    AVG(share_price_change) AS average_change
FROM SharePrices
GROUP BY strftime('%Y-%m-%d %H', trade_time)
ORDER BY hourly_bucket;


-- By Minute of the Hour (combines all hours)
SELECT
    strftime('%M', trade_time) AS minute,
    COUNT(*) AS number_of_trades
FROM SharePrices
GROUP BY strftime('%M', trade_time)
ORDER BY minute;


-- By Individual Minute
SELECT
    strftime('%Y-%m-%d %H:%M', trade_time) AS minute_bucket,
    AVG(share_price_change) AS average_change
FROM SharePrices
GROUP BY strftime('%Y-%m-%d %H:%M', trade_time)
ORDER BY minute_bucket;