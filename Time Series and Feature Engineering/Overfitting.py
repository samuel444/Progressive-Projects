import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Create toy data
np.random.seed(42)

x = np.linspace(-3, 3, 40)
y = np.sin(x) + np.random.normal(0, 0.2, size=len(x))


# Keep the same observations in every train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)


degrees = [1,2,3,5,7,10,15]
results = []

x_curve = np.linspace(x.min(), x.max(), 500)


for degree in degrees:

    # Vandermonde matrices:
    # [1, x, x², ..., x^degree]
    X_train = np.vander(
        x_train,
        N=degree + 1,
        increasing=True
    )

    X_test = np.vander(
        x_test,
        N=degree + 1,
        increasing=True
    )

    X_curve = np.vander(
        x_curve,
        N=degree + 1,
        increasing=True
    )

    # The constant column is already in the Vandermonde matrix,
    # so sklearn should not add another intercept
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    curve_predictions = model.predict(X_curve)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_predictions)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, test_predictions)
    )

    results.append({
        "degree": degree,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse
    })

    plt.figure(figsize=(9, 5))

    plt.scatter(
        x_train,
        y_train,
        label="Training data"
    )

    plt.scatter(
        x_test,
        y_test,
        marker="x",
        s=70,
        label="Test data"
    )

    plt.plot(
        x_curve,
        curve_predictions,
        label=f"Degree {degree}"
    )

    plt.title(
        f"Polynomial Degree {degree}\n"
        f"Train RMSE: {train_rmse:.3f} | "
        f"Test RMSE: {test_rmse:.3f}"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.show()


print("Degree | Train RMSE | Test RMSE")
print("--------------------------------")

for result in results:
    print(
        f"{result['degree']:>6} | "
        f"{result['train_rmse']:.4f}     | "
        f"{result['test_rmse']:.4f}"
    )


