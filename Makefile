.PHONY: setup portfolio credit test clean

PY ?= python3

setup:
	$(PY) -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e ".[dev]"

portfolio:
	.venv/bin/risklab portfolio

credit:
	.venv/bin/risklab credit

test:
	.venv/bin/pytest -q

clean:
	rm -rf reports .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
