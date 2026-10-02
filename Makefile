.PHONY: help setup data prepare test lint clean api sweep profile

help:
	@echo "make setup    - create venv and install dependencies"
	@echo "make data     - download the Kaggle dataset into data/raw"
	@echo "make prepare  - validate, split and scale; writes data/processed"
	@echo "make test     - run the test suite"
	@echo "make api      - run the scoring API locally"
	@echo "make sweep    - run the full technique x classifier sweep (hours)"
	@echo "make profile  - measure inference latency of every logged model"
	@echo "make clean    - remove caches and processed artefacts"

setup:
	python -m venv venv && ./venv/bin/pip install -U pip && ./venv/bin/pip install -r requirements.txt
	@echo "Activate with: source venv/bin/activate"

data:
	./scripts/get_data.sh

prepare:
	python src/data_loader.py --csv data/raw/creditcard.csv --out data/processed

test:
	pytest tests/ -v

api:
	uvicorn api.main:app --reload

sweep:
	python -m src.experiment_runner

profile:
	python -m src.inference_profiler

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache data/processed/*.npz data/processed/*.joblib
