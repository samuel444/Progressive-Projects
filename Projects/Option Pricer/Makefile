
.PHONY: install notebook test lint init-db build-cpp clean

install:
	python -m pip install -e ".[dev]"

notebook:
	jupyter lab notebooks/01_options_pricer_risk_engine.ipynb

test:
	pytest -q

lint:
	ruff check src tests
	black --check src tests

init-db:
	python scripts/init_database.py

build-cpp:
	bash scripts/build_cpp.sh

clean:
	rm -rf build cpp/build .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
