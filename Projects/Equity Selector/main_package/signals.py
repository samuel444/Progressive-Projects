import requests

########################################
# Telegram
########################################

BOT_TOKEN = "8640734587:AAERQyiff5doDpIa4gKp7RpCFmAWAS36lQ0"
CHAT_ID = 5156774786

def send_notification(message: str):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=10
    )

    response.raise_for_status()


import os
import requests
import yfinance as yf


########################################
# Trading 212
########################################

API_KEY = "20669136ZLUVmWtalekZrZwsbcZfvANKXoXDT"
API_SECRET = "8abFquzs3Ot8HHKo0mWoY6zezIxPKjEpHnoIbI2Gk4M"

BASE_URL = "https://demo.trading212.com/api/v0"

AUTH = (API_KEY, API_SECRET)


########################################
# Internal Helpers
########################################

def _as_list(tickers):

    if isinstance(tickers, str):
        return [tickers]

    return list(tickers)


def _get_account():

    response = requests.get(
        f"{BASE_URL}/equity/account/summary",
        auth=AUTH,
        timeout=10
    )

    response.raise_for_status()

    return response.json()


def _get_positions():

    response = requests.get(
        f"{BASE_URL}/equity/positions",
        auth=AUTH,
        timeout=10
    )

    response.raise_for_status()

    return response.json()


def _get_instruments():

    response = requests.get(
        f"{BASE_URL}/equity/metadata/instruments",
        auth=AUTH,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        return data.get("items", data.get("instruments", []))

    return data


def _resolve_t212_ticker(ticker):

    ticker = ticker.upper()

    # Already looks like a Trading 212 ticker
    if ticker.endswith("_EQ"):
        return ticker

    instruments = _get_instruments()

    matches = [
        instrument["ticker"]
        for instrument in instruments
        if instrument["ticker"].upper().startswith(
            ticker + "_"
        )
    ]

    # Prefer US listing for normal US symbols
    us_matches = [
        ticker
        for ticker in matches
        if "_US_EQ" in ticker
    ]

    if len(us_matches) == 1:
        return us_matches[0]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise ValueError(
            f"Could not find Trading 212 ticker for {ticker}"
        )

    raise ValueError(
        f"Multiple Trading 212 instruments found for "
        f"{ticker}: {matches}"
    )


def _get_price_in_account_currency(
    ticker,
    account_currency
):

    stock = yf.Ticker(ticker)

    info = stock.fast_info

    price = float(info.last_price)
    stock_currency = info.currency

    ########################################
    # Already Same Currency
    ########################################

    if stock_currency == account_currency:
        return price


    ########################################
    # Convert Currency
    ########################################

    fx_ticker = (
        f"{stock_currency}"
        f"{account_currency}=X"
    )

    try:

        fx = yf.Ticker(fx_ticker).fast_info.last_price

        return price * float(fx)

    except Exception:

        # Try inverse currency pair
        inverse_ticker = (
            f"{account_currency}"
            f"{stock_currency}=X"
        )

        fx = yf.Ticker(
            inverse_ticker
        ).fast_info.last_price

        return price / float(fx)


def _market_order(
    ticker,
    quantity,
    extended_hours=False
):

    t212_ticker = _resolve_t212_ticker(ticker)

    response = requests.post(
        f"{BASE_URL}/equity/orders/market",
        auth=AUTH,
        json={
            "ticker": t212_ticker,
            "quantity": quantity,
            "extendedHours": extended_hours
        },
        timeout=10
    )

    response.raise_for_status()

    return response.json()


########################################
# BUY
########################################

def buy_stock(
    tickers,
    amount,
    mode="cash",
    extended_hours=False
):

    tickers = _as_list(tickers)

    if amount <= 0:
        raise ValueError(
            "Amount must be greater than 0."
        )

    account = _get_account()

    currency = account["currency"]
    available_cash = account["cash"]["availableToTrade"]
    portfolio_value = account["totalValue"]


    ########################################
    # Determine Money Per Stock
    ########################################

    if mode == "cash":

        total_budget = amount

        if total_budget > available_cash:
            raise ValueError(
                f"Not enough cash. "
                f"Available: {available_cash:.2f} {currency}"
            )

        money_per_stock = (
            total_budget / len(tickers)
        )


    elif mode == "percent":

        if not 0 < amount <= 100:
            raise ValueError(
                "Percentage must be between 0 and 100."
            )

        total_budget = (
            portfolio_value
            * amount / 100
        )

        if total_budget > available_cash:
            raise ValueError(
                f"{amount}% of portfolio is "
                f"{total_budget:.2f} {currency}, "
                f"but only {available_cash:.2f} "
                f"{currency} is available."
            )

        money_per_stock = (
            total_budget / len(tickers)
        )


    elif mode == "shares":

        results = {}

        for ticker in tickers:

            results[ticker] = _market_order(
                ticker,
                amount,
                extended_hours
            )

        return results


    else:

        raise ValueError(
            "mode must be 'cash', "
            "'percent' or 'shares'"
        )


    ########################################
    # Convert Money -> Shares
    ########################################

    results = {}

    for ticker in tickers:

        price = _get_price_in_account_currency(
            ticker,
            currency
        )

        quantity = (
            money_per_stock / price
        )

        quantity = round(
            quantity,
            8
        )

        results[ticker] = _market_order(
            ticker,
            quantity,
            extended_hours
        )


    return results


########################################
# SELL
########################################

def sell_stock(
    tickers,
    amount,
    mode="percent",
    extended_hours=False
):

    tickers = _as_list(tickers)

    if amount <= 0:
        raise ValueError(
            "Amount must be greater than 0."
        )

    account = _get_account()
    currency = account["currency"]

    positions = _get_positions()

    results = {}


    ########################################
    # Loop Through Stocks
    ########################################

    for ticker in tickers:

        t212_ticker = _resolve_t212_ticker(
            ticker
        )


        ########################################
        # Find Position
        ########################################

        position = None

        for p in positions:

            position_ticker = (
                p.get("ticker")
                or p.get(
                    "instrument",
                    {}
                ).get("ticker")
            )

            if position_ticker == t212_ticker:

                position = p
                break


        if position is None:

            raise ValueError(
                f"You do not own {ticker}."
            )


        owned_quantity = float(
            position["quantity"]
        )


        ########################################
        # Percentage
        ########################################

        if mode == "percent":

            if not 0 < amount <= 100:

                raise ValueError(
                    "Percentage must be "
                    "between 0 and 100."
                )

            quantity = (
                owned_quantity
                * amount / 100
            )


        ########################################
        # Cash
        ########################################

        elif mode == "cash":

            price = (
                _get_price_in_account_currency(
                    ticker,
                    currency
                )
            )

            quantity = amount / price

            if quantity > owned_quantity:

                raise ValueError(
                    f"You do not own enough "
                    f"{ticker} to sell "
                    f"{amount:.2f} {currency}."
                )


        ########################################
        # Shares
        ########################################

        elif mode == "shares":

            quantity = amount

            if quantity > owned_quantity:

                raise ValueError(
                    f"You only own "
                    f"{owned_quantity} shares "
                    f"of {ticker}."
                )


        else:

            raise ValueError(
                "mode must be 'cash', "
                "'percent' or 'shares'"
            )


        quantity = round(
            quantity,
            8
        )


        ########################################
        # Negative = Sell
        ########################################

        results[ticker] = _market_order(
            ticker,
            -quantity,
            extended_hours
        )


    return results
