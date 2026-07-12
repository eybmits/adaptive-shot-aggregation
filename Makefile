PYTHON ?= python

.PHONY: help test verify figures hardware paper release all clean

help:
	@printf '%s\n' \
	  'make test      Run implementation tests' \
	  'make verify    Verify result hashes, tables, figures, PDFs, and anonymity' \
	  'make figures   Rebuild all five manuscript figures from source evidence' \
	  'make hardware  Reanalyze archived IBM records and rebuild Fig. 5' \
	  'make paper     Build named and anonymous papers and supplements' \
	  'make release   Build and verify the anonymous source package' \
	  'make all       Run test, figures, paper, verify, and release' \
	  'make clean     Remove local caches and TeX intermediates'

test:
	$(PYTHON) -m pytest -q

verify:
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
	rm -f output/pdf/split_ensemble_qrc_*.pdf
	cp paper/main.pdf output/pdf/QCE26_NIER_split_ensemble_qrc_named.pdf
	cp paper/main_anon.pdf output/pdf/QCE26_NIER_split_ensemble_qrc_anonymous.pdf
	cp paper/supplement.pdf output/pdf/QCE26_NIER_split_ensemble_qrc_supplement.pdf
	cp paper/supplement_anon.pdf output/pdf/QCE26_NIER_split_ensemble_qrc_supplement_anonymous.pdf
	cd output/pdf && shasum -a 256 QCE26_NIER_split_ensemble_qrc_named.pdf QCE26_NIER_split_ensemble_qrc_anonymous.pdf QCE26_NIER_split_ensemble_qrc_supplement.pdf QCE26_NIER_split_ensemble_qrc_supplement_anonymous.pdf > SHA256SUMS

release:
	$(PYTHON) scripts/package_anonymous_release.py
	$(PYTHON) release/qce2026_nier_anonymous/verify_release.py
	$(PYTHON) release/qce2026_nier_anonymous/verify_release.py --build

all: test figures paper verify release

clean:
	rm -rf .pytest_cache scripts/__pycache__ tests/__pycache__ paper/.pytest_cache
	cd paper && latexmk -c main.tex && latexmk -c main_anon.tex && latexmk -c supplement.tex && latexmk -c supplement_anon.tex
	cd paper/gfx && latexmk -c fig1_shot_organization.tex
