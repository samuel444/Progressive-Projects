SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    cot.total_quantity,
    cot.number_of_orders
FROM Customers AS c
INNER JOIN (

    SELECT
        customer_id,
        SUM(quantity) AS total_quantity,
        COUNT(*) AS number_of_orders
    FROM Orders
    GROUP BY customer_id

) AS cot
ON c.customer_id = cot.customer_id
WHERE cot.total_quantity > 5
ORDER BY cot.total_quantity DESC;



WITH CustomerOrderTotals AS (

    SELECT
        customer_id,
        SUM(quantity) AS total_quantity,
        COUNT(*) AS number_of_orders
    FROM Orders
    GROUP BY customer_id

)

SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    cot.total_quantity,
    cot.number_of_orders
FROM Customers AS c
INNER JOIN CustomerOrderTotals AS cot
ON c.customer_id = cot.customer_id
WHERE cot.total_quantity > 5
ORDER BY cot.total_quantity DESC;

