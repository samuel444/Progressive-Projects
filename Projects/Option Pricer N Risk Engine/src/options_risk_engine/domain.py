
"""Domain objects used across pricing, forecasting and risk modules."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

@dataclass
class OptionTicker:
    """Store all market, forecast and option-chain state for one ticker.

    The object is deliberately a state holder rather than a pricing engine.
    Pricing, validation and simulation remain in separate functions so they
    can be tested independently.
    """

    symbol: str
    target_dte: int = 45
    risk_free_rate: float = 0.0375
    dividend_yield: float = 0.0
    forecast_lookback: int = 1260
    buy_edge: float = 0.05
    parity_tolerance: float = 1e-6
    valuation_date: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp.today().normalize()
    )

    forecast_horizon: int = field(init=False)
    current_price: Optional[float] = None
    expiry: Optional[pd.Timestamp] = None
    calendar_dte: Optional[int] = None
    trading_dte: Optional[int] = None
    time_to_expiry: Optional[float] = None
    forward_price: Optional[float] = None
    discounted_spot: Optional[float] = None

    volatility_inputs: dict[str, float] = field(default_factory=dict)
    call_chain: Optional[pd.DataFrame] = field(default=None, repr=False)
    put_chain: Optional[pd.DataFrame] = field(default=None, repr=False)

    call_analysis: Optional[pd.DataFrame] = field(default=None, repr=False)
    put_analysis: Optional[pd.DataFrame] = field(default=None, repr=False)
    parity_table: Optional[pd.DataFrame] = field(default=None, repr=False)
    monte_carlo_result: Optional[dict[str, Any]] = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.valuation_date = pd.Timestamp(self.valuation_date).normalize()

        if not self.symbol:
            raise ValueError("Ticker symbol cannot be empty")
        if self.target_dte <= 0:
            raise ValueError("target_dte must be positive")
        if self.risk_free_rate <= -1:
            raise ValueError("risk_free_rate is invalid")
        if self.dividend_yield < 0:
            raise ValueError("dividend_yield cannot be negative")
        if self.forecast_lookback <= 0:
            raise ValueError("forecast_lookback must be positive")

        self.forecast_horizon = self._trading_days_to_target()

    def _trading_days_to_target(self) -> int:
        target_date = (
            self.valuation_date
            + pd.Timedelta(days=self.target_dte)
        )

        return max(
            int(
                np.busday_count(
                    self.valuation_date.date(),
                    target_date.date(),
                )
            ),
            1,
        )

    @property
    def target_date(self) -> pd.Timestamp:
        return (
            self.valuation_date
            + pd.Timedelta(days=self.target_dte)
        )

    @property
    def expiry_string(self) -> str:
        if self.expiry is None:
            raise ValueError(f"{self.symbol} does not have an expiry")
        return self.expiry.strftime("%Y-%m-%d")

    @property
    def call_record(self) -> tuple[pd.DataFrame, str]:
        if self.call_chain is None:
            raise ValueError(f"{self.symbol} does not have a call chain")
        return self.call_chain, self.expiry_string

    @property
    def put_record(self) -> tuple[pd.DataFrame, str]:
        if self.put_chain is None:
            raise ValueError(f"{self.symbol} does not have a put chain")
        return self.put_chain, self.expiry_string

    def attach_option_chains(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        expiry: str,
    ) -> None:
        if not isinstance(calls, pd.DataFrame) or calls.empty:
            raise ValueError(f"{self.symbol} call chain is empty or invalid")
        if not isinstance(puts, pd.DataFrame) or puts.empty:
            raise ValueError(f"{self.symbol} put chain is empty or invalid")

        self.call_chain = calls.copy()
        self.put_chain = puts.copy()
        self.expiry = pd.Timestamp(expiry).normalize()
        self.refresh_derived_market_values()

    def set_current_price(self, current_price: float) -> None:
        current_price = float(current_price)
        if not np.isfinite(current_price) or current_price <= 0:
            raise ValueError(
                f"{self.symbol} current price must be finite and positive"
            )

        self.current_price = current_price
        self.refresh_derived_market_values()

    def set_volatility_inputs(
        self,
        volatility_inputs: Mapping[str, float],
    ) -> None:
        converted = {
            name: float(value)
            for name, value in volatility_inputs.items()
        }

        invalid = {
            name: value
            for name, value in converted.items()
            if not np.isfinite(value) or value <= 0
        }

        if invalid:
            raise ValueError(
                f"Invalid volatility inputs for {self.symbol}: {invalid}"
            )

        self.volatility_inputs = converted

    def refresh_derived_market_values(self) -> None:
        if self.expiry is not None:
            self.calendar_dte = max(
                (self.expiry - self.valuation_date).days,
                0,
            )
            self.trading_dte = max(
                int(
                    np.busday_count(
                        self.valuation_date.date(),
                        self.expiry.date(),
                    )
                ),
                0,
            )
            self.time_to_expiry = self.calendar_dte / 365.0

        if (
            self.current_price is not None
            and self.time_to_expiry is not None
        ):
            self.discounted_spot = (
                self.current_price
                * np.exp(-self.dividend_yield * self.time_to_expiry)
            )
            self.forward_price = (
                self.current_price
                * np.exp(
                    (
                        self.risk_free_rate
                        - self.dividend_yield
                    )
                    * self.time_to_expiry
                )
            )

    def ensure_pricing_ready(self) -> None:
        missing = []

        if self.call_chain is None:
            missing.append("call_chain")
        if self.put_chain is None:
            missing.append("put_chain")
        if self.expiry is None:
            missing.append("expiry")
        if self.current_price is None:
            missing.append("current_price")
        if self.time_to_expiry is None:
            missing.append("time_to_expiry")

        if missing:
            raise ValueError(
                f"{self.symbol} is not ready for pricing; missing {missing}"
            )

    def table_key(self, option_type: str) -> str:
        return f"{self.symbol} {option_type}"
