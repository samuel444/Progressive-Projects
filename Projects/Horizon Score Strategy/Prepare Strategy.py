from equity_selector.config import data_root as model_data_root
from horizon_score.config import data_root as project_data_root

"""Run model preparation first; switch phase to selection/final after preceding stages finish."""

from horizon_score.preparation import prepare_research

SETTINGS = {
    "phase": "selection",  # model, selection, final
    "model_dir": str(model_data_root()),
    "selection_dir": str(project_data_root() / "portfolio_selection"),
    "final_dir": str(project_data_root() / "final_evaluation"),
    "download_fx": True,
}

if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    print(prepare_research(**SETTINGS))
