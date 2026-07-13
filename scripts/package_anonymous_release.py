#!/usr/bin/env python3
"""Create, compile, checksum, and scrub the isolated anonymous paper release."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
DESTINATION = ROOT / "release" / "qce2026_nier_anonymous"
SOURCE_DATE_EPOCH = "1783641600"
PAPER_TITLE = "From Quantum Shots to Training Data: Reorganizing Measurement Records for Improved Finite-Shot Quantum Learning"

SECTION_FILES = (
    "abstract.tex",
    "introduction.tex",
    "background.tex",
    "related-work.tex",
    "methodology.tex",
    "experimental-section.tex",
    "evaluation.tex",
    "conclusion.tex",
)

FIGURE_FILES = (
    "fig2_composite.pdf",
    "fig3_regime_map.pdf",
    "fig4_controls_compact.pdf",
    "fig5_hardware.pdf",
)

CONFIG_FILES = (
    "benchmark.json",
    "qai_robust_protocol.json",
    "qai_robust_protocol_santa_fe.json",
)

PROTOCOL_DOC_FILES = ()

DATA_FILES = (
    "sunspots_monthly_sidc.csv",
    "santa_fe_laser_A.dat",
    "santa_fe_laser_A_continuation.dat",
)

EXPRESSIVE_RESULT_FILES = (
    "ideal_diagnostics.json",
    "ideal_diagnostics_seed_2718.json",
    "ideal_diagnostics_seed_31415.json",
    "ideal_feature_grid.csv",
)

ROBUST_RESULT_FILES = (
    "manifest.json",
    "replicate_results.csv",
    "split_validation_spectrum.csv",
    "cluster_summary.csv",
    "cluster_paired_stats.csv",
    "ideal_vs_classical.csv",
    "partition_robustness_summary.csv",
    "fixed_lane_sensitivity.csv",
    "selected_k_frequency.csv",
    "generated/main_table_body.tex",
    "generated/result_macros_candidate.tex",
    "generated/report_manifest.json",
    "generated/summary_go_no_go.json",
)

MANUSCRIPT_RESULT_FILES = (
    "manifest.json",
    "replicate_results.csv",
    "split_validation_spectrum.csv",
    "cluster_summary.csv",
    "cluster_paired_stats.csv",
    "ideal_vs_classical.csv",
    "partition_robustness_summary.csv",
    "fixed_lane_sensitivity.csv",
    "selected_k_frequency.csv",
)

REPRODUCTION_SCRIPTS = (
    "base_impl.py",
    "expressive_reservoir.py",
    "robust_protocol.py",
    "robust_reservoir_adapter.py",
    "run_expressive_reservoir_diagnostics.py",
    "run_robust_evaluation.py",
    "protocol_config.py",
    "report_qai_robust_results.py",
    "plot_manuscript_figures.py",
    "plot_hardware_figure.py",
    "analyze_qai_hardware.py",
    "build_santa_fe_manuscript_results.py",
)

IDENTIFYING_LITERALS = (
    "Markus Baumann",
    "Maximilian Zorn",
    "Thomas Gabor",
    "Claudia Linnhoff-Popien",
    "Jonas Stein",
    "markus.baumann@campus.lmu.de",
    "QAR-Lab",
    "LMU Munich",
    "University of Exeter",
    "github.com/eybmits",
    "eybmits",
    "/Users/",
    "0009-0007-3575-1006",
    "0009-0006-2750-7495",
    "0000-0003-2048-8667",
    "0000-0001-6284-9286",
    "0000-0001-5727-9151",
    "LMU Sustainability Fund",
    "Munich Quantum Valley",
    "Bavarian StMWi",
)

README = """# Anonymous QCE26 NIER source release

This directory is a standalone, identity-redacted source release for double-blind review.
The six-page main paper uses the eight-section source layout; each released section
is copied from the current paper and the identity-bearing conclusion branches are replaced by
an anonymous artifact placeholder. No named source or stale rewrite is included. The five-page
supplement, frozen configurations, source and merged result tables, completed Santa hardware
extension evidence, raw hardware records, and scripts needed to validate and redraw the reported
figures retain repository-relative paths.

The anonymous artifact address is intentionally a placeholder in
`anonymous_artifact_url.tex`. Replace only that macro when the anonymous review address is
available.

## Verify and rebuild

Requirements: Python 3, `latexmk`, and Poppler (`pdfinfo` and `pdftotext`).

```sh
python3 verify_release.py
python3 verify_release.py --build
```

The checksums, anonymous boundary, PDF contract, and isolated rebuild can be
rechecked without rerunning either campaign:

```sh
python3 verify_release.py --build
```

With NumPy, pandas, SciPy, and Matplotlib installed, the main-paper figures
can be redrawn with `PYTHONDONTWRITEBYTECODE=1 python3 scripts/plot_manuscript_figures.py`.
The archived hardware panel can be redrawn with
`PYTHONDONTWRITEBYTECODE=1 python3 scripts/plot_hardware_figure.py`. These commands write
to `paper/figures/`; the release's immutable submitted PDFs remain in `figures/` and are
covered by `SHA256SUMS`.

The build check compiles both documents from a temporary copy, verifies byte-for-byte PDF
reproducibility, scans source, extracted PDF text, PDF metadata, and printable PDF strings,
and leaves no auxiliary files in this directory.

`SHA256SUMS` covers every released file except the checksum list itself. The package must
contain no auxiliary build traces or version-control metadata.
"""

ARTIFACT_PLACEHOLDER = r"""% Replace only this macro after receiving the anonymous review address.
\providecommand{\AnonymousArtifactURL}{\textit{[anonymous artifact URL to be supplied]}}
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _replace_author_block(text: str, *, label: str) -> str:
    """Replace the named/anonymous author conditional with anonymous-only source."""

    title_at = text.find("\\title{")
    start = text.find("\\ifdefined\\ANON", title_at)
    end = text.find("\\maketitle", start)
    if title_at < 0 or start < 0 or end < 0:
        raise RuntimeError(f"{label}: author block anchors were not found")
    block = text[start:end]
    if block.count("\\ifdefined\\ANON") != 1 or block.count("\\fi") != 1:
        raise RuntimeError(f"{label}: author block is not the expected single conditional")
    anonymous = (
        "\\hypersetup{pdfauthor={}}\n"
        "\\author{\\IEEEauthorblockN{Anonymous Authors}}\n\n"
    )
    return text[:start] + anonymous + text[end:]


def _insert_artifact_macro(text: str, *, label: str) -> str:
    anchor = "\\urlstyle{same}\n"
    if text.count(anchor) != 1:
        raise RuntimeError(f"{label}: expected one URL-style insertion anchor")
    return text.replace(anchor, anchor + "\\input{anonymous_artifact_url.tex}\n", 1)


def sanitized_main() -> str:
    """Create the anonymous equivalent of the live restored ``paper/main.tex``."""

    text = (PAPER / "main.tex").read_text(encoding="utf-8")
    if PAPER_TITLE not in text:
        raise RuntimeError("main manuscript title differs from the restored paper title")
    text = _replace_author_block(text, label="main")
    text = text.replace("\\usepackage{orcidlink}\n", "")
    if text.count("\\balance") != 1:
        raise RuntimeError(
            "main: expected one bibliography-balance command"
        )
    if "\\input{sections/conclusion.tex}" not in text:
        # Backwards-compatible fail-closed handling if the restored source is
        # temporarily flattened again.
        artifact_pattern = re.compile(
            r"\\ifdefined\\ANON\\else\s*The named artifact is available.*?\\fi",
            re.DOTALL,
        )
        text, count = artifact_pattern.subn(
            r"The anonymous artifact will be available at \\AnonymousArtifactURL.",
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"main: expected one named artifact branch, replaced {count}")
        acknowledgment_pattern = re.compile(
            r"\\ifdefined\\ANON\\else\s*\\section\*\{Acknowledgment\}.*?\\fi",
            re.DOTALL,
        )
        text, count = acknowledgment_pattern.subn("", text, count=1)
        if count != 1:
            raise RuntimeError(f"main: expected one acknowledgment branch, replaced {count}")
    text = _insert_artifact_macro(text, label="main")
    text = re.sub(
        r"^% QCE26 NIER short paper\..*?^\\documentclass",
        lambda _: "% Anonymous-only QCE26 NIER short paper.\n\\documentclass",
        text,
        count=1,
        flags=re.DOTALL | re.MULTILINE,
    )
    if "\\ifdefined" in text or "\\orcidlink" in text or "mailto:" in text:
        raise RuntimeError("main: conditional or identifying author machinery remains")
    return text


def sanitized_experimental_section() -> str:
    """Copy the setup after confirming that it contains no identifiers."""

    text = (PAPER / "sections" / "experimental-section.tex").read_text(
        encoding="utf-8"
    )
    if "\\ifdefined" in text or "github.com" in text.casefold():
        raise RuntimeError("setup: identifying conditional or public repository URL remains")
    return text


def sanitized_conclusion() -> str:
    """Copy the current conclusion while removing any named-only branches."""

    text = (PAPER / "sections" / "conclusion.tex").read_text(encoding="utf-8")
    acknowledgment_pattern = re.compile(
        r"\\ifdefined\\ANON\\else\s*\\section\*\{Acknowledgment\}.*?\\fi",
        re.DOTALL,
    )
    # Acknowledgments are optional in the working manuscript.  Remove the
    # named-only block when present, but do not require authors to include one.
    text = acknowledgment_pattern.sub("", text, count=1)
    if "\\ifdefined" in text or "github.com" in text.casefold():
        raise RuntimeError("conclusion: identifying conditional or public repository URL remains")
    return text


def sanitized_supplement() -> str:
    text = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    if PAPER_TITLE not in text:
        raise RuntimeError("supplement title differs from the restored paper title")
    text = _replace_author_block(text, label="supplement")
    artifact_pattern = re.compile(
        r"The named artifact is available at\s*\\ifdefined\\ANON.*?\\fi",
        re.DOTALL,
    )
    text, count = artifact_pattern.subn(
        r"The anonymous artifact will be available at \\AnonymousArtifactURL.", text, count=1
    )
    if count != 1:
        raise RuntimeError(f"expected one supplement artifact branch, replaced {count}")
    text = _insert_artifact_macro(text, label="supplement")
    # Remove named/anonymous wrapper build comments: this file is anonymous-only.
    text = re.sub(
        r"^% Standalone supplementary material.*?^\\documentclass",
        lambda _: "% Anonymous-only standalone supplementary material.\n\\documentclass",
        text,
        count=1,
        flags=re.DOTALL | re.MULTILINE,
    )
    if "\\ifdefined" in text or "\\orcidlink" in text or "mailto:" in text:
        raise RuntimeError("supplement: conditional or identifying author machinery remains")
    return text


def assert_no_identifying_literals(root: Path) -> None:
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() == ".pdf":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for literal in IDENTIFYING_LITERALS:
            if literal.casefold() in text.casefold():
                findings.append(f"{path.relative_to(root)} contains {literal!r}")
    if findings:
        raise RuntimeError("anonymous source scrub failed:\n" + "\n".join(findings))


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"required release input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RuntimeError(f"required release directory is missing: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.name not in {".DS_Store"}:
            copy_file(path, destination / path.relative_to(source))


def populate_release() -> None:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    (DESTINATION / "gfx").mkdir(parents=True)
    (DESTINATION / "figures").mkdir()
    (DESTINATION / "sections").mkdir()

    write_text(DESTINATION / "main.tex", sanitized_main())
    write_text(DESTINATION / "supplement.tex", sanitized_supplement())
    for name in SECTION_FILES:
        if name == "experimental-section.tex":
            write_text(DESTINATION / "sections" / name, sanitized_experimental_section())
        elif name == "conclusion.tex":
            write_text(DESTINATION / "sections" / name, sanitized_conclusion())
        else:
            copy_file(PAPER / "sections" / name, DESTINATION / "sections" / name)
    write_text(DESTINATION / "anonymous_artifact_url.tex", ARTIFACT_PLACEHOLDER)
    write_text(DESTINATION / "README.md", README)
    for name in (
        "IEEEtran.cls",
        "IEEEtranDoi.bst",
        "bstcontrol.bib",
        "main_table_body.tex",
        "references.bib",
        "results_macros.tex",
    ):
        copy_file(PAPER / name, DESTINATION / name)
    copy_file(
        PAPER / "gfx" / "fig1_shot_organization.pdf",
        DESTINATION / "gfx" / "fig1_shot_organization.pdf",
    )
    for name in FIGURE_FILES:
        copy_file(PAPER / "figures" / name, DESTINATION / "figures" / name)

    for name in CONFIG_FILES:
        copy_file(ROOT / "configs" / name, DESTINATION / "configs" / name)
    for name in PROTOCOL_DOC_FILES:
        copy_file(ROOT / "docs" / name, DESTINATION / "docs" / name)
    for name in DATA_FILES:
        copy_file(ROOT / "data" / name, DESTINATION / "data" / name)
    for name in EXPRESSIVE_RESULT_FILES:
        copy_file(
            ROOT / "results" / "expressive_reservoir" / name,
            DESTINATION / "results" / "expressive_reservoir" / name,
        )
    for name in ROBUST_RESULT_FILES:
        copy_file(
            ROOT / "results" / "qai_robust_protocol_full" / name,
            DESTINATION / "results" / "qai_robust_protocol_full" / name,
        )
    for directory in ("qai_santa_fe_laser_screen", "qai_robust_protocol_santa_fe_manuscript"):
        for name in MANUSCRIPT_RESULT_FILES:
            copy_file(
                ROOT / "results" / directory / name,
                DESTINATION / "results" / directory / name,
            )
    copy_tree(ROOT / "results" / "qai_hardware", DESTINATION / "results" / "qai_hardware")
    copy_tree(
        ROOT / "results" / "qai_hardware_santa_fe_extension",
        DESTINATION / "results" / "qai_hardware_santa_fe_extension",
    )
    for name in REPRODUCTION_SCRIPTS:
        copy_file(ROOT / "scripts" / name, DESTINATION / "scripts" / name)

    copy_file(
        Path(__file__).with_name("anonymous_release_verify.py"),
        DESTINATION / "verify_release.py",
    )
    assert_no_identifying_literals(DESTINATION)


def compile_pdfs() -> None:
    env = os.environ.copy()
    env.update(
        {
            "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
            "FORCE_SOURCE_DATE": "1",
            "TZ": "UTC",
        }
    )
    with tempfile.TemporaryDirectory(prefix="anonymous-release-package-") as temporary:
        build_root = Path(temporary) / "source"
        shutil.copytree(DESTINATION, build_root)
        for stem in ("main", "supplement"):
            result = subprocess.run(
                [
                    "latexmk",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"{stem}.tex",
                ],
                cwd=build_root,
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
            )
            if result.returncode:
                raise RuntimeError(
                    f"isolated {stem} build failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
                )
            shutil.copy2(build_root / f"{stem}.pdf", DESTINATION / f"{stem}.pdf")


def write_checksums() -> None:
    files = sorted(path for path in DESTINATION.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256(path)}  {path.relative_to(DESTINATION)}" for path in files]
    write_text(DESTINATION / "SHA256SUMS", "\n".join(lines) + "\n")


def verify_release() -> None:
    subprocess.run(
        [sys.executable, str(DESTINATION / "verify_release.py"), "--root", str(DESTINATION)],
        cwd=DESTINATION,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(DESTINATION / "verify_release.py"),
            "--root",
            str(DESTINATION),
            "--build",
        ],
        cwd=DESTINATION,
        check=True,
    )


def main() -> None:
    populate_release()
    compile_pdfs()
    write_checksums()
    verify_release()
    print(f"anonymous source release ready: {DESTINATION}")
    for name in ("main.pdf", "supplement.pdf", "SHA256SUMS"):
        print(f"{sha256(DESTINATION / name)}  {name}")


if __name__ == "__main__":
    main()
