from .returns import all_return_features
from .momentum import all_momentum_features
from .volatility import all_volatility_features
from .range_volatility import all_range_volatility_features
from .trend import all_trend_features
from .moving_averages import all_moving_average_features
from .drawdown import all_drawdown_features
from .distribution import all_distribution_features
from .tail_risk import all_tail_risk_features
from .volume import all_volume_features
from .liquidity import all_liquidity_features
from .ohlc import all_ohlc_features
from .market_relative import all_market_relative_features
from .sector_relative import all_sector_relative_features
from .beta import all_beta_features
from .correlation import all_correlation_features
from .residual import all_residual_features
from .cross_sectional import all_cross_sectional_features
from .breadth import all_breadth_features
from .dispersion import all_dispersion_features
from .technical import all_technical_features
from .interactions import all_interaction_features
from .composite import all_composite_features
from .regimes import all_regime_features
from .experimental import all_experimental_features


FEATURE_GROUPS = {
    "returns": all_return_features,
    "momentum": all_momentum_features,
    "volatility": all_volatility_features,
    "range_volatility": all_range_volatility_features,
    "trend": all_trend_features,
    "moving_averages": all_moving_average_features,
    "drawdown": all_drawdown_features,
    "distribution": all_distribution_features,
    "tail_risk": all_tail_risk_features,
    "volume": all_volume_features,
    "liquidity": all_liquidity_features,
    "ohlc": all_ohlc_features,
    "market_relative": all_market_relative_features,
    "sector_relative": all_sector_relative_features,
    "beta": all_beta_features,
    "correlation": all_correlation_features,
    "residual": all_residual_features,
    "cross_sectional": all_cross_sectional_features,
    "breadth": all_breadth_features,
    "dispersion": all_dispersion_features,
    "technical": all_technical_features,
    "interactions": all_interaction_features,
    "composite": all_composite_features,
    "regimes": all_regime_features,
    "experimental": all_experimental_features,
}
