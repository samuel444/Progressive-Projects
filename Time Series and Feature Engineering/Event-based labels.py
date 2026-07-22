import yfinance as yf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

df = df[["Close"]]
df["Return"] = df["Close"].pct_change()

df = df.dropna()

profit = 0.05
loss = 0.05

labels = []

max_holding_period = 20

for i in range(len(df)):

    start_price = df["Close"].values[i]
    label = np.nan

    for future_price in df["Close"].values[i+1:i+max_holding_period+1]:

        change = future_price / start_price - 1

        if change >= profit:
            label = 1
            break

        if change <= -loss:
            label = 0
            break

    labels.append(label)

df["Label"] = labels


df["Predictor"] = df["Return"].rolling(window=20).mean()

df = df.dropna()
length = len(df)

actual_values = []
predicted_values = []

for i in range(int(length * 0.25), length - 10, 10):

    train = df.iloc[:i-max_holding_period]
    test = df.iloc[i:i + 10]

    X_train = train[["Predictor"]].values
    y_train = train["Label"].values

    X_test = test[["Predictor"]].values
    y_test = test["Label"].values

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    actual_values.extend(y_test)
    predicted_values.extend(predictions)

actual_values = np.array(actual_values)
predicted_values = np.array(predicted_values)

accuracy = accuracy_score(actual_values, predicted_values)
precision = precision_score(actual_values, predicted_values)
recall = recall_score(actual_values, predicted_values)
f1 = f1_score(actual_values, predicted_values)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
