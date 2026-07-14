#!/usr/bin/env python3
"""Verify the Santa Fe manuscript tables, provenance, and PDF boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "qai_robust_protocol_santa_fe_manuscript"
HARDWARE = ROOT / "results" / "qai_hardware"
HARDWARE_EXTENSION = ROOT / "results" / "qai_hardware_santa_fe_extension"
PAPER = ROOT / "paper" if (ROOT / "paper").is_dir() else ROOT
TASKS = ("mackey_glass", "santa_fe_laser", "rossler", "sunspots")
PAPER_TITLE = (
    "From Quantum Shots to Training Data: "
    "Reorganizing Measurement Records in Quantum Machine Learning"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"[ok] {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_pages(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"pdfinfo did not report a page count for {path}")


def pdf_title(path: Path) -> str:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in output.splitlines():
        if line.startswith("Title:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"pdfinfo did not report a title for {path}")


def pdf_text(path: Path) -> str:
    return subprocess.check_output(["pdftotext", str(path), "-"], text=True)


def main() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "complete-manuscript-evidence-merge",
            "manifest records the complete manuscript evidence merge")
    require(tuple(manifest["tasks"]) == TASKS, "manifest has the four manuscript tasks")

    for name, metadata in manifest["outputs"].items():
        path = RESULTS / name
        require(path.exists(), f"{name} exists")
        require(sha256(path) == metadata["sha256"], f"{name} hash matches manifest")
        rows = pd.read_csv(path)
        require(len(rows) == int(metadata["rows"]), f"{name} row count matches manifest")
        require(not rows.duplicated().any(), f"{name} has no duplicate rows")
        require(sorted(rows["dataset"].unique()) == metadata["tasks"],
                f"{name} has the recorded task set")

    paired = pd.read_csv(RESULTS / "cluster_paired_stats.csv")
    headline = paired.loc[
        paired["budget_setting"].eq("balanced")
        & paired["reservoir_variant"].eq("expressive_reuploading")
        & paired["feature_set"].eq("z_plus_zz")
        & paired["baseline"].eq("ev")
        & paired["method"].eq("split")
    ].set_index("dataset")
    expected = {
        "mackey_glass": (0.04454345493644963, 0.037175879047682034, 0.051911030825217225),
        "santa_fe_laser": (0.084787, 0.017418, 0.152156),
        "rossler": (0.034926, 0.013947, 0.055904),
        "sunspots": (0.035954, 0.010328, 0.061580),
    }
    require(set(headline.index) == set(TASKS), "balanced paired table covers every task")
    for task, values in expected.items():
        observed = headline.loc[task, [
            "delta_baseline_minus_method", "cluster_ci95_low", "cluster_ci95_high"
        ]].to_numpy(dtype=float)
        require(np.allclose(observed, values, atol=5e-7, rtol=0.0),
                f"{task} balanced paired effect matches admitted results")

    hardware_expected = {
        HARDWARE: (0.749831905668201, 0.7114384875458315,
                   0.03839341812236948, 0.011473806548939256),
        HARDWARE / "marrakesh": (0.7440068094248644, 0.7126115085886292,
                                 0.03139530083623522, 0.007965138504288939),
    }
    for directory, values in hardware_expected.items():
        summary = pd.read_csv(directory / "hardware_summary.csv")
        paired_hw = pd.read_csv(directory / "hardware_paired_stats.csv")
        require(set(summary["dataset"]) == set(TASKS),
                f"{directory.name} hardware summary has the four manuscript tasks")
        require(set(paired_hw["dataset"]) == set(TASKS),
                f"{directory.name} hardware paired table has the four manuscript tasks")
        santa_summary = summary[summary["dataset"].eq("santa_fe_laser")].set_index("protocol")
        santa_paired = paired_hw[
            paired_hw["dataset"].eq("santa_fe_laser")
            & paired_hw["baseline"].eq("ev")
            & paired_hw["method"].eq("split")
        ].iloc[0]
        observed = np.array([
            santa_summary.loc["ev", "mean"],
            santa_summary.loc["split", "mean"],
            santa_paired["delta_mean"],
            santa_paired["delta_ci95"],
        ], dtype=float)
        require(np.allclose(observed, values, atol=5e-12, rtol=0.0),
                f"{directory.name} Santa hardware result matches admitted values")
        require(int(santa_paired["n"]) == 20 and float(santa_paired["win_rate"]) == 0.95,
                f"{directory.name} Santa result contains twenty blocks and nineteen wins")

    extension = json.loads((HARDWARE_EXTENSION / "manifest.json").read_text(encoding="utf-8"))
    require(extension["confirmed_usage_seconds"] == 111.0,
            "Santa hardware extension records 111 charged QPU seconds")
    require(extension["jobs"]["ibm_fez"]["job_id"] == "d99mf4t2su3c739kl9k0"
            and extension["jobs"]["ibm_marrakesh"]["job_id"] == "d99mf52f47jc73a99fj0",
            "Santa hardware extension records both admitted IBM jobs")

    main_section_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PAPER / "sections").glob("*.tex"))
    )
    manuscript_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            PAPER / "supplement.tex",
            PAPER / "results_macros.tex",
            PAPER / "main_table_body.tex",
        ]
    )
    manuscript_sources = main_section_sources + "\n" + manuscript_sources
    compact_sources = " ".join(manuscript_sources.split())
    main_source = (PAPER / "main.tex").read_text(encoding="utf-8")
    abstract_source = (PAPER / "sections" / "abstract.tex").read_text(encoding="utf-8")
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", abstract_source, re.DOTALL
    )
    require(abstract_match is not None, "abstract environment exists")
    abstract_body = re.sub(
        r"\\(?:emph|textbf|textit)\{([^}]*)\}", r"\1", abstract_match.group(1)
    )
    abstract_body = abstract_body.replace(r'\"o', "ö").replace(r"\_", "_")
    abstract_body = re.sub(r"\\[A-Za-z]+", " ", abstract_body)
    abstract_body = abstract_body.replace("{", " ").replace("}", " ")
    abstract_words = re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", abstract_body)
    require(len(abstract_words) <= 180,
            f"abstract contains at most 180 words ({len(abstract_words)})")
    require(re.search(r"\d", abstract_body) is None,
            "abstract contains no numerical values")
    abstract_protocol_terms = ("validation", "development fold", "selected", "tuned jointly")
    require(not any(term in abstract_body.casefold() for term in abstract_protocol_terms),
            "abstract omits model-selection protocol details")
    figure_count = len(re.findall(r"\\begin\{figure\*?\}", main_section_sources))
    figure_labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", main_section_sources))
    figure_refs = set(re.findall(r"\\ref\{(fig:[^}]+)\}", main_section_sources))
    require(figure_count == len(figure_labels),
            "every main-paper figure has one unique label")
    require(figure_labels.issubset(figure_refs),
            "every main-paper figure is referenced in the prose")
    standard_sections = (
        "Introduction",
        "Related Work",
        "Method",
        "Experimental Setup",
        "Evaluation",
        "Discussion",
        "Conclusion",
    )
    for title in standard_sections:
        require(rf"\section{{{title}}}" in main_section_sources,
                f"main paper contains the standard {title} section")
    require("Focused Evaluation" not in main_section_sources
            and "Emergent Results" not in main_section_sources
            and "Discussion and Practical Use" not in main_section_sources,
            "nonstandard section titles are absent")
    require(r"\input{results_macros.tex}" in main_source,
            "manuscript imports generated result macros")
    require(r"\input{main_table_body.tex}" in main_source,
            "manuscript imports the generated headline table")
    require("EVSplit" not in manuscript_sources and "EVGrouped" in manuscript_sources,
            "internal result macros use the current Grouped terminology")
    headline_table = (PAPER / "main_table_body.tex").read_text(encoding="utf-8")
    require("Lorenz" not in headline_table,
            "removed task is absent from the active headline table")
    require("Santa Fe" in manuscript_sources, "Santa Fe is present in manuscript sources")
    normalized_main_sections = " ".join(main_section_sources.casefold().split())
    require("NIER" not in main_section_sources,
            "main text does not use the submission category as scientific framing")
    dense_implementation_terms = re.compile(r"\b(?:row|rows|matrix|matrices|nested)\b", re.I)
    require(dense_implementation_terms.search(main_section_sources) is None,
            "main text avoids implementation-centric row and matrix language")
    require("adaptive shot aggregation" in compact_sources.casefold(),
            "adaptive shot aggregation terminology is explicit")
    require("validation-selected hyperparameter of the classical readout"
            in compact_sources.casefold(),
            "approved validation-selected hyperparameter wording is present")
    obsolete_tuning_terms = (
        "selects the group size",
        "selected group size",
        "selecting the group size by validation",
        "aggregation scale as a validation parameter",
    )
    require(not any(term in compact_sources.casefold() for term in obsolete_tuning_terms),
            "obsolete group-size selection terminology is absent")
    require("six-qubit" in compact_sources and "ten-qubit" in compact_sources,
            "simulation and hardware circuit sizes are explicit")
    require("not a hardware implementation of the six-qubit simulator" in compact_sources,
            "hardware transfer is distinguished from the simulation reservoir")
    require(r"\documentclass[10pt,conference]{IEEEtran}" in main_source
            and "a4paper" not in main_source,
            "manuscript uses the IEEE conference class declaration")
    require(not any(term in manuscript_sources for term in ("NO-GO", "post-hoc", "replaces Lorenz")),
            "main and supplement avoid obsolete decision-gate framing")
    obsolete_hardware = (
        "qai_same_day_hardware_confirmation",
        "preferred twelve-round",
        "fallback eight-round",
        "600-second cap",
        "separate four-task hardware protocol",
        "backend rounds",
    )
    require(not any(term in manuscript_sources for term in obsolete_hardware),
            "manuscript excludes the superseded future hardware campaign")

    named_path = PAPER / "main.pdf"
    anonymous_path = PAPER / "main_anon.pdf"
    require(pdf_pages(named_path) == 6, "named manuscript has six pages")
    require(pdf_pages(anonymous_path) == 6, "anonymous manuscript has six pages")
    require(pdf_title(named_path) == PAPER_TITLE, "named PDF has the current title")
    require(pdf_title(anonymous_path) == PAPER_TITLE, "anonymous PDF has the current title")
    named_text = pdf_text(named_path)
    anon_text = pdf_text(anonymous_path)
    require("Santa Fe" in named_text and "Lorenz" not in named_text,
            "named PDF contains the coherent four-task benchmark")
    explicit_reservoir_seeds = ("1729", "2718", "31415")
    require(not any(seed in named_text for seed in explicit_reservoir_seeds)
            and not any(seed in anon_text for seed in explicit_reservoir_seeds),
            "main manuscripts omit explicit reservoir seed identifiers")
    hardware_figure_text = pdf_text(PAPER / "figures" / "fig5_hardware.pdf")
    require("Santa Fe" in hardware_figure_text and "Lorenz" not in hardware_figure_text,
            "hardware figure contains the completed four-task suite")
    forbidden = ("Markus Baumann", "Maximilian Zorn", "Thomas Gabor",
                 "Claudia Linnhoff-Popien", "Jonas Stein", "campus.lmu.de",
                 "github.com/eybmits")
    require(not any(token in anon_text for token in forbidden), "anonymous PDF passes privacy scan")
    legacy_terms = (
        "Reorganizing Quantum Measurement Records",
        "Split ensemble",
        "SPLIT",
        "Covariance-EV",
    )
    require(not any(token in named_text for token in legacy_terms),
            "named PDF uses the current visible method terminology")
    require(not any(token in anon_text for token in legacy_terms),
            "anonymous PDF uses the current visible method terminology")
    print("Santa Fe manuscript package verified.")


if __name__ == "__main__":
    main()
