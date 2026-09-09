"""Create the centrally configured model research data directory."""
from equity_selector.config import data_root

if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    data_root().mkdir(parents=True, exist_ok=True)
    print(data_root())
