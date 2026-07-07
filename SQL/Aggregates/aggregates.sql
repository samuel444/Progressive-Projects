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