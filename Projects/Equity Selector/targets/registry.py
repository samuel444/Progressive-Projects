from .returns import all_return_targets
from .volatility import all_volatility_targets
from .direction import all_direction_targets
from .barriers import all_barrier_targets
from .excursions import all_excursion_targets
from .drawdown import all_drawdown_targets
from .risk_adjusted import all_risk_adjusted_targets
from .ranking import all_ranking_targets


TARGET_GROUPS = {
    "returns": all_return_targets,
    "volatility": all_volatility_targets,
    "direction": all_direction_targets,
    "barriers": all_barrier_targets,
    "excursions": all_excursion_targets,
    "drawdown": all_drawdown_targets,
    "risk_adjusted": all_risk_adjusted_targets,
    "ranking": all_ranking_targets,
}
