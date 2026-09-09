"""Fit pending model-only requests; this stage performs no strategy backtesting."""

from equity_selector.artifacts import produce_requests

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-dir")
    args = parser.parse_args()
    for path in produce_requests(args.requests_dir):
        print(path)
