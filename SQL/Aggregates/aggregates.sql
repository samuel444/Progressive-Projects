SELECT *
FROM Orders;

SELECT COUNT(*)
FROM Orders;

SELECT SUM(quantity)
FROM Orders;

SELECT AVG(quantity)
FROM Orders;

SELECT MAX(quantity)
FROM Orders;

SELECT MIN(quantity)
FROM Orders;


SELECT
customer_id,
SUM(quantity)
FROM Orders
GROUP BY customer_id;

SELECT
customer_id,
MAX(quantity)
FROM Orders
GROUP BY customer_id;

SELECT
product_id,
customer_id,
SUM(quantity)
FROM Orders
GROUP BY
product_id,
customer_id;


SELECT
customer_id,
SUM(quantity) AS total_quantity
FROM Orders
GROUP BY customer_id
ORDER BY total_quantity ASC;

