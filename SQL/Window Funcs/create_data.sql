CREATE TABLE Customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    age INTEGER,
    country TEXT
);


INSERT INTO Customers
VALUES
(1,'Alice','Smith',25,'UK'),
(2,'Bob','Jones',31,'USA'),
(3,'Charlie','Brown',28,'Canada'),
(4,'David','Taylor',42,'UK'),
(5,'Emily','White',35,'Australia');


CREATE TABLE Products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    price REAL
);


INSERT INTO Products
VALUES
(1,'Laptop','Electronics',999.99),
(2,'Keyboard','Electronics',45.50),
(3,'Mouse','Electronics',20.99),
(4,'Desk','Furniture',180.00),
(5,'Chair','Furniture',120.00);


CREATE TABLE Orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    order_date TEXT
);


INSERT INTO Orders
VALUES
(1,1,1,1,'2026-01-05'),
(2,2,2,2,'2026-01-06'),
(3,1,3,1,'2026-01-10'),
(4,4,5,4,'2026-01-15'),
(5,5,4,1,'2026-01-18'),
(6,3,2,3,'2026-01-20'),
(7,2,1,3,'2026-01-21'),
(8,2,2,4,'2026-01-18'),
(9,2,1,3,'2026-01-07'),
(10,4,3,2,'2026-01-23');


SELECT * FROM Customers;

SELECT * FROM Products;

SELECT * FROM Orders;