SELECT
customer_id,
SUM(quantity)
FROM Orders
GROUP BY customer_id;


-- SUM() OVER(PARTITION BY) - Total for each customer
SELECT
    order_id,
    customer_id,
    quantity,
    SUM(quantity) OVER (
        PARTITION BY customer_id
    ) AS customer_total
FROM Orders;


-- AVG() OVER(PARTITION BY)
SELECT
    order_id,
    customer_id,
    quantity,
    AVG(quantity) OVER (
        PARTITION BY customer_id
    ) AS average_quantity
FROM Orders;


-- COUNT() OVER(PARTITION BY)
SELECT
    order_id,
    customer_id,
    quantity,
    COUNT(*) OVER (
        PARTITION BY customer_id
    ) AS number_of_orders
FROM Orders;


-- ROW_NUMBER()
SELECT
    order_id,
    customer_id,
    quantity,
    ROW_NUMBER() OVER (
        PARTITION BY customer_id
    ) AS row_number
FROM Orders;


-- RANK()
SELECT
    order_id,
    customer_id,
    quantity,
    RANK() OVER (
        PARTITION BY customer_id
        ORDER BY quantity DESC
    ) AS customer_rank
FROM Orders;


-- DENSE_RANK()
SELECT
    order_id,
    customer_id,
    quantity,
    DENSE_RANK() OVER (
        PARTITION BY customer_id
        ORDER BY quantity DESC
    ) AS dense_customer_rank
FROM Orders;


-- LAG()
SELECT
    order_id,
    customer_id,
    quantity,
    LAG(quantity) OVER (
        PARTITION BY customer_id
    ) AS previous_quantity
FROM Orders;


-- LEAD()
SELECT
    order_id,
    customer_id,
    quantity,
    LEAD(quantity) OVER (
        PARTITION BY customer_id
    ) AS next_quantity
FROM Orders;