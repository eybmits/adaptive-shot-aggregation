#!/usr/bin/env python3
"""Build and verify the named arXiv source archive."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
DESTINATION = ROOT / "release" / "arxiv_source.zip"
PAPER_TITLE = (
    "From Quantum Shots to Training Data: "
    "Reorganizing Measurement Records in Quantum Machine Learning"
)
SOURCE_DATE_EPOCH = "1783987200"

SECTION_FILES = (
    "abstract.tex",
    "introduction.tex",
    "related-work.tex",
    "background.tex",
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
ROOT_FILES = (
    "IEEEtran.cls",
    "IEEEtranDoi.bst",
    "bstcontrol.bib",
    "main_table_body.tex",
    "references.bib",
    "results_macros.tex",
)
FORBIDDEN = (
    "Anonymous Authors",
    "double-blind",
    "QCE26",
    "NIER",
    "ChatGPT",
    "OpenAI",
    "Codex",
    "placeholder",
    "/Users/",
)
BUILD_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contains_publication_title(text: str) -> bool:
    """Compare the title after removing TeX-only line breaks and whitespace."""

    normalized = " ".join(text.replace("\\\\", " ").split())
    return PAPER_TITLE in normalized


def named_main_source() -> str:
    """Return the main source with the unused anonymous branch removed."""

    text = (PAPER / "main.tex").read_text(encoding="utf-8")
    pattern = re.compile(
        r"\\ifdefined\\ANON\s*.*?\\else\s*(.*?)\\fi",
        flags=re.DOTALL,
    )
    text, count = pattern.subn(r"\1", text, count=1)
    if count != 1:
        raise RuntimeError("main.tex does not contain the expected author conditional")
    if not contains_publication_title(text):
        raise RuntimeError("main.tex title does not match the publication title")
    return text


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"required arXiv input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def populate(stage: Path) -> None:
    (stage / "sections").mkdir(parents=True)
    (stage / "figures").mkdir()
    (stage / "gfx").mkdir()
    (stage / "main.tex").write_text(named_main_source(), encoding="utf-8")
    for name in ROOT_FILES:
        copy_required(PAPER / name, stage / name)
    for name in SECTION_FILES:
        copy_required(PAPER / "sections" / name, stage / "sections" / name)
    for name in FIGURE_FILES:
        copy_required(PAPER / "figures" / name, stage / "figures" / name)
    copy_required(
        PAPER / "gfx" / "fig1_shot_organization.pdf",
        stage / "gfx" / "fig1_shot_organization.pdf",
    )


def build(stage: Path, command: list[str]) -> None:
    env = os.environ.copy()
    env.update(
        {
            "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
            "FORCE_SOURCE_DATE": "1",
            "TZ": "UTC",
        }
    )
    result = subprocess.run(
        command,
        cwd=stage,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(
            f"arXiv source build failed: {' '.join(command)}\n"
            f"{result.stdout[-5000:]}\n{result.stderr[-5000:]}"
        )


def scan(stage: Path) -> None:
    findings: list[str] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".pdf", ".bbl"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in FORBIDDEN:
            if token.casefold() in text.casefold():
                findings.append(f"{path.relative_to(stage)} contains {token!r}")
    if findings:
        raise RuntimeError("arXiv source scrub failed:\n" + "\n".join(findings))


def clean_build_files(stage: Path) -> None:
    for path in stage.iterdir():
        if not path.is_file():
            continue
        if path.name == "main.pdf" or any(path.name.endswith(suffix) for suffix in BUILD_SUFFIXES):
            path.unlink()


def write_zip(stage: Path) -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        DESTINATION.unlink()
    fixed_time = (2026, 7, 14, 0, 0, 0)
    with zipfile.ZipFile(DESTINATION, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(relative, fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def verify_archive() -> None:
    with zipfile.ZipFile(DESTINATION) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise RuntimeError(f"corrupt member in arXiv archive: {broken}")
        names = set(archive.namelist())
        if "main.tex" not in names or "main.bbl" not in names:
            raise RuntimeError("arXiv archive lacks main.tex or main.bbl")
        if any(Path(name).suffix in BUILD_SUFFIXES or name.endswith("main.pdf") for name in names):
            raise RuntimeError("arXiv archive contains local build products")

    with tempfile.TemporaryDirectory(prefix="arxiv-source-verify-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(DESTINATION) as archive:
            archive.extractall(root)
        for _ in range(3):
            build(root, ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
        info = subprocess.check_output(["pdfinfo", str(root / "main.pdf")], text=True)
        if "Pages:           6" not in info or f"Title:           {PAPER_TITLE}" not in info:
            raise RuntimeError("isolated arXiv build does not match the six-page paper contract")
        log = (root / "main.log").read_text(encoding="utf-8", errors="replace")
        bad = ("undefined references", "Citation `", "LaTeX Error", "Overfull \\hbox")
        if any(token in log for token in bad):
            raise RuntimeError("isolated arXiv build contains unresolved references or layout errors")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="arxiv-source-stage-") as temporary:
        stage = Path(temporary)
        populate(stage)
        build(stage, ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
        if not (stage / "main.bbl").is_file():
            raise RuntimeError("latexmk did not create main.bbl")
        scan(stage)
        clean_build_files(stage)
        write_zip(stage)
    verify_archive()
    print(f"arXiv source ready: {DESTINATION}")
    print(f"sha256: {sha256(DESTINATION)}")


if __name__ == "__main__":
    main()
