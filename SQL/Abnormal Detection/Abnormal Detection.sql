-- Find abnormal hourly average changes
SELECT
    strftime('%Y-%m-%d %H', trade_time) AS hourly_bucket,
    AVG(share_price_change) AS average_change,
    CASE
        WHEN ABS(AVG(share_price_change)) > 1.0 THEN 'Abnormal'
        ELSE 'Normal'
    END AS change_status
FROM SharePrices
GROUP BY strftime('%Y-%m-%d %H', trade_time)
ORDER BY hourly_bucket;


-- Flag hours where the average change is more than three times the overall average
SELECT
    strftime('%Y-%m-%d %H', trade_time) AS hourly_bucket,
    AVG(share_price_change) AS average_change,
    CASE
        WHEN ABS(AVG(share_price_change)) >
             3 * ABS(
                 (
                     SELECT AVG(share_price_change)
                     FROM SharePrices
                 )
             )
        THEN 'Abnormal'
        ELSE 'Normal'
    END AS change_status
FROM SharePrices
GROUP BY strftime('%Y-%m-%d %H', trade_time)
ORDER BY hourly_bucket;


-- Use a rolling window to compare each hour against the previous 7 hours
WITH HourlyChanges AS (

    SELECT
        strftime('%Y-%m-%d %H', trade_time) AS hourly_bucket,
        AVG(share_price_change) AS average_change
    FROM SharePrices
    GROUP BY strftime('%Y-%m-%d %H', trade_time)

)

SELECT
    hourly_bucket,
    average_change,
    AVG(average_change) OVER (
        ORDER BY hourly_bucket
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS rolling_7_hour_average,
    CASE
        WHEN  1.2 < 
            ABS(average_change - (AVG(average_change) OVER (
                 ORDER BY hourly_bucket
                 ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
             )))
        THEN 'Abnormal'
        ELSE 'Normal'
    END AS change_status
FROM HourlyChanges
ORDER BY hourly_bucket;
