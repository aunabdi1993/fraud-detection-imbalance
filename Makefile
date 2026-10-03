.PHONY: help setup data prepare export test lint clean api

help:
	@echo "make setup    - create venv and install dependencies"
	@echo "make data     - download the Kaggle dataset into data/raw"
	@echo "make prepare  - validate, split and scale; writes data/processed"
	@echo "make export   - fit and package the model bundle the API serves"
	@echo "make test     - run the test suite"
	@echo "make api      - run the scoring API locally"
	@echo "make clean    - remove caches and processed artefacts"

setup:
	python -m venv venv && ./venv/bin/pip install -U pip && ./venv/bin/pip install -r requirements.txt
	@echo "Activate with: source venv/bin/activate"

data:
	./scripts/get_data.sh

prepare:
	python src/data_loader.py --csv data/raw/creditcard.csv --out data/processed

export:
	python -m src.export_model --model xgboost

test:
	pytest tests/ -v

api:
	uvicorn api.main:app --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache data/processed/*.npz data/processed/*.joblib
