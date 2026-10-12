PYTHON ?= python
SOURCE_DATE_EPOCH ?= 1783987200
FORCE_SOURCE_DATE ?= 1
TZ ?= UTC
export SOURCE_DATE_EPOCH FORCE_SOURCE_DATE TZ

.PHONY: help test check verify-data verify figures hardware paper release arxiv all clean

help:
	@printf '%s\n' \
	  'make test      Run implementation tests' \
	  'make check     Run tests and frozen-table checks (Python only)' \
	  'make verify-data Verify frozen table hashes, row counts, and source manifests' \
	  'make verify    Verify result hashes, tables, figures, PDFs, and anonymity' \
	  'make figures   Rebuild all five manuscript figures from source evidence' \
	  'make hardware  Reanalyze archived IBM records and rebuild Fig. 5' \
	  'make paper     Build named and anonymous papers and supplements' \
	  'make release   Build and verify the anonymous review package' \
	  'make arxiv     Build and verify the named arXiv source ZIP' \
	  'make all       Run tests and rebuild every publication artifact' \
	  'make clean     Remove local caches and TeX intermediates'

test:
	$(PYTHON) -m pytest -q

check: test verify-data

verify-data:
	$(PYTHON) scripts/build_santa_fe_manuscript_results.py --validate-only

verify: verify-data
	$(PYTHON) scripts/verify_santa_fe_manuscript.py

figures:
	cd paper/gfx && latexmk -pdf -interaction=nonstopmode -halt-on-error fig1_shot_organization.tex
	$(PYTHON) scripts/plot_manuscript_figures.py --fig 2
	$(PYTHON) scripts/plot_manuscript_figures.py --fig 3
	$(PYTHON) scripts/plot_manuscript_figures.py --fig 4
	$(PYTHON) scripts/plot_hardware_figure.py

hardware:
	$(PYTHON) scripts/analyze_qai_hardware.py
	$(PYTHON) scripts/analyze_qai_hardware.py --subdir marrakesh
	$(PYTHON) scripts/plot_hardware_figure.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_anon.tex
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement_anon.tex
	mkdir -p output/pdf
	rm -f output/pdf/*.pdf output/pdf/SHA256SUMS
	cp paper/main.pdf output/pdf/adaptive_shot_aggregation.pdf
	cp paper/main_anon.pdf output/pdf/adaptive_shot_aggregation_anonymous.pdf
	cp paper/supplement.pdf output/pdf/adaptive_shot_aggregation_supplement.pdf
	cp paper/supplement_anon.pdf output/pdf/adaptive_shot_aggregation_supplement_anonymous.pdf
	cd output/pdf && shasum -a 256 adaptive_shot_aggregation.pdf adaptive_shot_aggregation_anonymous.pdf adaptive_shot_aggregation_supplement.pdf adaptive_shot_aggregation_supplement_anonymous.pdf > SHA256SUMS

release:
	$(PYTHON) scripts/package_anonymous_release.py
	$(PYTHON) release/adaptive_shot_aggregation_anonymous_review/verify_release.py
	$(PYTHON) release/adaptive_shot_aggregation_anonymous_review/verify_release.py --build

arxiv:
	$(PYTHON) scripts/package_arxiv_source.py

all: test hardware figures paper verify release arxiv

clean:
	rm -rf .pytest_cache scripts/__pycache__ tests/__pycache__ paper/.pytest_cache
	cd paper && latexmk -c main.tex && latexmk -c main_anon.tex && latexmk -c supplement.tex && latexmk -c supplement_anon.tex
	cd paper/gfx && latexmk -c fig1_shot_organization.tex
