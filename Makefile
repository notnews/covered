# covered -- reproducible source-concentration analysis of CNN transcripts.
# Each data target reads from one data/ stage and writes the next; all are
# idempotent and parquet/checksum-cached, so re-running is cheap.

PY := uv run
PILOT_START ?= 2014-06-18
PILOT_END   ?= 2015-12-31

.PHONY: setup lint format typecheck test check \
        acquire parse extract resolve hhi validate figures pilot all clean

setup:  ## create venv, install package + dev/llm extras, download spaCy model
	uv venv --python 3.12
	uv pip install -e ".[dev,llm]"
	$(PY) python -m spacy download en_core_web_lg
	$(PY) python -m spacy download en_core_web_trf

lint:  ## ruff lint
	$(PY) ruff check src tests

format:  ## ruff format + import sort
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

typecheck:  ## mypy
	$(PY) mypy

test:  ## pytest on fixtures (no data download)
	$(PY) pytest

check: lint typecheck test  ## everything CI runs

acquire:  ## download the 8 Dataverse CSVs (needs DATAVERSE_API_TOKEN)
	$(PY) covered acquire

parse:  ## provenance + speaker-turn parse -> interim/turns.parquet
	$(PY) covered parse

extract:  ## inline quote-attribution -> interim/attributions.parquet
	$(PY) covered extract

resolve:  ## entity resolution -> processed/entities.parquet
	$(PY) covered resolve

hhi:  ## annual HHI series -> outputs/tables/hhi_annual.csv
	$(PY) covered hhi

validate:  ## stratified LLM validation + error-adjusted HHI
	$(PY) covered validate

figures:  ## HHI time-series plots + coverage diagnostic
	$(PY) covered figures

pilot:  ## end-to-end on one year w/ face-validity assertions
	$(PY) covered pilot --start $(PILOT_START) --end $(PILOT_END)

all: acquire parse extract resolve hhi figures  ## full 25-year run

clean:  ## remove interim/processed caches (keeps raw + reference)
	rm -rf data/interim/* data/processed/* outputs/figures/*
