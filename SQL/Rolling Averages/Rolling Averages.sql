-- Return of the week, ending current day
SELECT
    day,
    share_price_change,
    SUM(share_price_change) OVER (
        ORDER BY day
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS last_week_return
FROM SharePrices;


-- Return of the week, starting current day
SELECT
    day,
    share_price_change,
    SUM(share_price_change) OVER (
        ORDER BY day
        ROWS BETWEEN CURRENT ROW AND 6 FOLLOWING
    ) AS next_week_return
FROM SharePrices;


-- Return of the previous 3 days, current day, and next 3 days
SELECT
    day,
    share_price_change,
    SUM(share_price_change) OVER (
        ORDER BY day
        ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
    ) AS return_previous_3_current_next_3
FROM SharePrices;


-- Return of the running total from the first day to the current day
SELECT
    day,
    share_price_change,
    SUM(share_price_change) OVER (
        ORDER BY day
        ROWS BETWEEN UNBOUNDED PRECEDING AND PRECEDING 1
    ) AS running_return
FROM SharePrices;

