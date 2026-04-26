.PHONY: help install dev-install lint format test ingest index eval api ui clean

help:
	@echo "make install        — install runtime dependencies"
	@echo "make dev-install    — install + pre-commit hooks"
	@echo "make lint           — ruff check + mypy"
	@echo "make format         — ruff format + auto-fix"
	@echo "make test           — pytest (unit only)"
	@echo "make test-all       — pytest including integration markers"
	@echo "make ingest         — download SEC filings"
	@echo "make index          — full pipeline: download → parse → chunk → embed"
	@echo "make eval           — run RAGAS evaluation"
	@echo "make api            — run FastAPI dev server"
	@echo "make ui             — run Streamlit app"
	@echo "make clean          — remove caches and processed data"

install:
	pip install -r requirements.txt

dev-install: install
	pre-commit install

lint:
	ruff check src tests app scripts
	mypy src

format:
	ruff format src tests app scripts
	ruff check --fix src tests app scripts

test:
	pytest -m "not integration"

test-all:
	pytest

ingest:
	python -c "from src.ingest.downloader import download_filings; download_filings()"

index:
	python scripts/build_index.py

eval:
	python -m src.evaluation.evaluate

api:
	uvicorn src.api.main:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf data/processed/* chroma_db
