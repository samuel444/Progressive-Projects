SELECT *
FROM Customers
INNER JOIN Orders 
ON Customers.customer_id = Orders.customer_id;


SELECT *
FROM Customers AS c
LEFT JOIN Orders AS o
ON c.customer_id = o.customer_id;


SELECT *
FROM Customers AS c
RIGHT JOIN Orders AS o
ON c.customer_id = o.customer_id;


SELECT *
FROM Orders AS o
LEFT JOIN Products AS p
ON o.product_id = p.product_id;


SELECT
    c.first_name,
    p.product_name,
    o.quantity
FROM Customers AS c
INNER JOIN Orders AS o
ON c.customer_id = o.customer_id
INNER JOIN Products AS p
ON o.product_id = p.product_id;


DROP TABLE IF EXISTS joinedTable;

CREATE TABLE joinedTable AS
SELECT
    c.first_name,
    p.product_name,
    o.quantity
FROM Customers AS c
INNER JOIN Orders AS o
ON c.customer_id = o.customer_id
INNER JOIN Products AS p
ON o.product_id = p.product_id;