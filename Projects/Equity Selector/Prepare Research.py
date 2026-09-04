"""Run model preparation first; switch phase to selection/final after preceding stages finish."""

from equity_selector.preparation import prepare_research

SETTINGS = {
    "phase": "model",  # model, selection, final
    "model_dir": "data/extensive_20260904/model_research",
    "selection_dir": "data/extensive_20260904/portfolio_selection",
    "final_dir": "data/extensive_20260904/final_evaluation",
    "download_fx": True,
}

if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    print(prepare_research(**SETTINGS))
