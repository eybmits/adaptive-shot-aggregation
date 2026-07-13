#!/usr/bin/env python3
"""Fail-closed verifier for the standalone anonymous source release.

This file is copied into the release package as ``verify_release.py``.  It
contains no identifying names.  Known identifying phrases are represented by
one-way hashes and checked after conservative text normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


SOURCE_DATE_EPOCH = "1783641600"
PAPER_TITLE = "From Raw Shots to Expectation Values: Shot Grouping Improves Time Series Prediction"
PDF_CONTRACT = {
    "main.pdf": {
        "pages": 6,
        "title": PAPER_TITLE,
    },
    "supplement.pdf": {
        "pages": 5,
        "title": f"Supplementary Material: {PAPER_TITLE}",
    },
}
LETTER_WIDTH_PT = 612.0
LETTER_HEIGHT_PT = 792.0
PAGE_TOLERANCE_PT = 1.0
EXPECTED_SECTIONS = {
    "sections/abstract.tex",
    "sections/introduction.tex",
    "sections/background.tex",
    "sections/related-work.tex",
    "sections/methodology.tex",
    "sections/experimental-section.tex",
    "sections/evaluation.tex",
    "sections/conclusion.tex",
}
BUILD_TRACE_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
    ".toc",
}
TEXT_SUFFIXES = {".bib", ".bst", ".cls", ".csv", ".json", ".md", ".py", ".tex", ".txt"}
REQUIRED_FILES = {
    "IEEEtran.cls",
    "IEEEtranDoi.bst",
    "README.md",
    "SHA256SUMS",
    "anonymous_artifact_url.tex",
    "bstcontrol.bib",
    "configs/qai_robust_protocol.json",
    "configs/qai_robust_protocol_santa_fe.json",
    "configs/benchmark.json",
    "data/santa_fe_laser_A.dat",
    "data/santa_fe_laser_A_continuation.dat",
    "data/sunspots_monthly_sidc.csv",
    "figures/fig2_composite.pdf",
    "gfx/fig1_shot_organization.pdf",
    "figures/fig3_regime_map.pdf",
    "figures/fig4_controls_compact.pdf",
    "figures/fig5_hardware.pdf",
    "main.pdf",
    "main.tex",
    "main_table_body.tex",
    "references.bib",
    "results_macros.tex",
    "results/expressive_reservoir/ideal_diagnostics.json",
    "results/expressive_reservoir/ideal_diagnostics_seed_2718.json",
    "results/expressive_reservoir/ideal_diagnostics_seed_31415.json",
    "results/expressive_reservoir/ideal_feature_grid.csv",
    "results/qai_robust_protocol_full/cluster_paired_stats.csv",
    "results/qai_robust_protocol_full/cluster_summary.csv",
    "results/qai_robust_protocol_full/fixed_lane_sensitivity.csv",
    "results/qai_robust_protocol_full/generated/main_table_body.tex",
    "results/qai_robust_protocol_full/generated/report_manifest.json",
    "results/qai_robust_protocol_full/generated/result_macros_candidate.tex",
    "results/qai_robust_protocol_full/generated/summary_go_no_go.json",
    "results/qai_robust_protocol_full/ideal_vs_classical.csv",
    "results/qai_robust_protocol_full/manifest.json",
    "results/qai_robust_protocol_full/partition_robustness_summary.csv",
    "results/qai_robust_protocol_full/replicate_results.csv",
    "results/qai_robust_protocol_full/selected_k_frequency.csv",
    "results/qai_robust_protocol_full/split_validation_spectrum.csv",
    "scripts/analyze_qai_hardware.py",
    "scripts/base_impl.py",
    "scripts/build_santa_fe_manuscript_results.py",
    "scripts/expressive_reservoir.py",
    "scripts/plot_hardware_figure.py",
    "scripts/plot_manuscript_figures.py",
    "scripts/protocol_config.py",
    "scripts/report_qai_robust_results.py",
    "scripts/robust_protocol.py",
    "scripts/robust_reservoir_adapter.py",
    "scripts/run_expressive_reservoir_diagnostics.py",
    "scripts/run_robust_evaluation.py",
    *EXPECTED_SECTIONS,
    "supplement.pdf",
    "supplement.tex",
    "verify_release.py",
}

for _prefix in (
    "results/qai_santa_fe_laser_screen",
    "results/qai_robust_protocol_santa_fe_manuscript",
):
    for _name in (
        "manifest.json",
        "replicate_results.csv",
        "split_validation_spectrum.csv",
        "cluster_summary.csv",
        "cluster_paired_stats.csv",
        "ideal_vs_classical.csv",
        "partition_robustness_summary.csv",
        "fixed_lane_sensitivity.csv",
        "selected_k_frequency.csv",
    ):
        REQUIRED_FILES.add(f"{_prefix}/{_name}")

for _prefix in ("results/qai_hardware", "results/qai_hardware/marrakesh"):
    REQUIRED_FILES.update(
        {
            f"{_prefix}/hardware_ablation.csv",
            f"{_prefix}/hardware_noise_diagnostics.json",
            f"{_prefix}/hardware_paired_stats.csv",
            f"{_prefix}/hardware_seed_level.csv",
            f"{_prefix}/hardware_spectrum.csv",
            f"{_prefix}/hardware_summary.csv",
            f"{_prefix}/layout.json",
            f"{_prefix}/manifest.json",
        }
    )
    for _task in ("mackey_glass", "santa_fe_laser", "rossler", "sunspots"):
        REQUIRED_FILES.add(f"{_prefix}/inputs_{_task}.npz")
        REQUIRED_FILES.add(f"{_prefix}/raw_{_task}.npz")

REQUIRED_FILES.update(
    {
        "results/qai_hardware_santa_fe_extension/manifest.json",
    }
)

# Each pair is (word count, sha256 of normalized lower-case phrase).  The
# phrases cover all manuscript-author names, identifying affiliations, and
# acknowledgment-only fund/project labels.  Hashes avoid putting the identity
# itself back into the anonymous package's checker.
FORBIDDEN_PHRASE_HASHES = {
    (1, "a737c69af72b5a8a48c030a4bbb7145b137c0afd3b8857cdd16ea252fdb4e1c3"),
    (1, "0778a3475d7e78269098e10a9cbf975d4b92184d49c5c03e1403f915c8855732"),
    (1, "da345b68e49a2136e2dff4a41c96045907060196afc7db660e17b1a3a3cffab2"),
    (1, "d545425d6a8dd1d8d20fe4727b9bf25534bcd00a86017b2ebc460ed9f4e07cb0"),
    (1, "df76ca6c399bccfe403998b14e4848a6b212c714d62d9bfa6ce9f1d7e8bd0c8d"),
    (1, "a13458e8c9f0b690501c4ca19c4ca99d147a2a04ab1348a8b4cde70ee5c06f2c"),
    (1, "411c19f57274b51fe8e9fc34e133743f931fba306ceb53743d2f03958aa33a1a"),
    (1, "1996f83d90aa125bc6bf0c2c2915f2492cf90fbc3059d9d2f96cdf3e1d463b7f"),
    (1, "c702f2e723810a1646f4d55ffec4612f79a6cea22a64e4bb81c3462aa851c1b4"),
    (1, "4bd3fc4d9f7c6aa36d63913a82d25ae341ba1e3477862b141bc8296220a5c95e"),
    (1, "ae976d383cbb3d7529b9c6292db44044571ffbaafbd8c516d6f1db04cdd7d15b"),
    (1, "723b80db62f087b5d9b52e8844cc0c27275eb21fd8a2429c0d9cbd00ed87cc15"),
    (1, "103c0764f5f779fd60aa890c95c2938a076f7d1fd8291d25d705af7cf114a0a5"),
    (1, "f952ae1e737e92a7842a54fbe61f2ca9ab9027ab6b5ebdad06f831cff1f5ce64"),
    (1, "3be7215bdd50f4866d57dd1db1511f6ef6a159968a238588c1b603bfa3eeb008"),
    (1, "aeb72d7f5a75373e3c3fc1f63420c85e081316c5799400b2d852bc56cd9f01ba"),
    (2, "414bb9fc04da21f1b9313a90f58bd091e18df4afbc80132690121839bd48e57f"),
    (2, "b6d13cf08a6c75d242669c22febfbd2cd15b02720ea1f5a438fa8412458da901"),
    (2, "8fc191e25b7e081d5d9365d691dca1a56ee6e3cf8ea48d2458f639d375469e52"),
    (2, "b81912ddc92b714340fe32311c88e2f5d84ce602ee2f6ac1f85854439576532f"),
    (2, "b87b76aeccf70f734189a00ab18c3ea6740d39e97f5c1e1a38e482c6d5fe4d48"),
    (2, "d986f98b301d5afc34fab2d60f37831be4858cfb980c0788e314e3a0255b3a86"),
    (2, "8ca18ef4c28a46733f607084bfe76865c4983ca1865c01efcd416f34a4e2cc6b"),
    (2, "b975e27c9925c94a6e231d0ec0118e9b2f4cf0adf17586f492d6fb4144d4ab7c"),
    (3, "24e175f89031bf85fd35ad2cc16df75d108b444f1fae6358a077486f6e885e07"),
    (3, "a3ba5840b5a4bf40bd87dc0f211487747ba3c184286a71cd3c6e118cc6fe96ae"),
    (3, "55fcacef2831ab1ed763f5326c10a7e307214961ab901318bb33c3ae49fc6d86"),
    (3, "c392aeec3218e0f8a2af62054d56f71bb76b4c468fb802a224681ed17dd615a7"),
    (1, "099ff8e8d814c2291fd2e0726b37ff659ae48d88d86f2711e10d8421701adbd0"),
    (1, "91af5af1b348a9a016a9d3b4730f9ec2392bfa108ae83a2ba1ab5221d166eebe"),
    (1, "3b0dcc1419123dbeee00b88326633dc45f238a40abc2179c709e3b7438c03686"),
    (1, "470b8720f9f865ed7e6f546bc2eec0b661f74d5244ee37cf54956761c12c589d"),
    (1, "ff65a849f406e84f7717100ce709ff70d8965e105c45066132f207b1faf54941"),
    (1, "e7e1fa7ca301efd2e042c5f208407363badbf066a3efae4d8a22b8282ee9656c"),
    (1, "e9daf3e4235e6aa3faa3364394c76ccc6d660912fc0f9d526d80289521d3cbba"),
    (1, "0b7cfe2b24f6bf14ee9e3e5241adb2f1f84f0c60e5e44adc3d36847ca4ba9a57"),
    (1, "2a6922ceca56f78d78dbb209b166443fd29aac14bab78dc75b5de214fdefee5f"),
    (1, "88bacb84548ef9f1c13f7e8eb6e81236fc0971c40372c1be6b813e847bb5065b"),
    (1, "899399e957cc15659c5a8b15dc3b3aab87ec6af162c269116a1c86b1d1b521bf"),
    (1, "b9eb29cb55b42c7ccd533a15225b71b9ec16892e74a4cdb7040bfe6e7d996713"),
    (1, "409e31bc4047a90602881cb2aef8b4d3e538557a637303f5a9eb98cdcdcaacbb"),
    (1, "b8f7bf180929c89904fd949e7333ba174c4bedbc6fffacb1c2fae7ba27443ac1"),
    (1, "6202b5b1b11ce9b36f82e75eedb39531e4d35164afc203eb6eed0b473397081c"),
    (2, "bd5c6f8d85f43b69e94dc7c86307d0fe274679ce882b8e1c560f7bae1f65ff23"),
    (2, "3dce908fc92c07e9fac14b1b30d137b0eccff3268d1645557d6d7df0dc6d47b1"),
    (2, "f47c58c7b18768b33ff9bc84859488625adcb721ac750eeff9b627cc5320d46e"),
    (2, "f89bd5a543be0f58d0789d84b1f1ffc0f7ccfb74a6207aef6edbb9e0b6c0240d"),
    (2, "2cd72b5be256e79a9807740ebe9d60a16a2e8e4fd9b355f4d0a27b9f11466bbc"),
    (2, "0a8a60c75da703d3b8ecf25e1572bdca53cca1eda966e844d1e1b2d205f55e94"),
    (2, "9e151ac48061d83518c4634c706a3b072564812485c62d3203070537b497e955"),
    (2, "3bf022b6ff76b982e63980a03be71ff1ed8fd5ed1ab1f02402187b3ac6b2dea5"),
    (3, "23336951743a99ae513ab74bd06faf69bc9d8d3c236cd5b443936c820eb4daa0"),
    (3, "369f8175eaa84de6cae43381f13dcecef42eda4b9049c78df7700b57491cbaf2"),
    (3, "dac35daddf44eac750ca91466d7b996e3531f7db9e4512cff077947a859f599c"),
    (3, "eb8f3ab6af173ac8c224dc595af10d9f01744f9f55a3eb4008316a9ecc1565ee"),
    (4, "d0e56ea33d9775c8b2a48964ace3b9ef6dbeb748d1ceb1fd547ea6e20f6d14e7"),
}

DIRECT_PATTERNS = {
    "email address": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "ORCID": re.compile(r"\b\d{4}-\d{4}-\d{4}-[\dX]{4}\b", re.I),
    "local user path": re.compile("/" + "Users" + "/", re.I),
    "public code-host URL": re.compile("github" + r"\." + "com", re.I),
    "local file URL": re.compile("file" + "://", re.I),
    "embedded working directory": re.compile(r"(?m)^PWD\s+/"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def hashed_phrase_hit(text: str, minimum_word_count: int = 1) -> bool:
    words = normalized_words(text)
    counts = sorted({
        count for count, _ in FORBIDDEN_PHRASE_HASHES
        if count >= minimum_word_count
    })
    for count in counts:
        targets = {digest for nwords, digest in FORBIDDEN_PHRASE_HASHES if nwords == count}
        for start in range(0, len(words) - count + 1):
            phrase = " ".join(words[start : start + count])
            if hashlib.sha256(phrase.encode("utf-8")).hexdigest() in targets:
                return True
    return False


def scan_text(label: str, text: str, findings: list[str]) -> None:
    for name, pattern in DIRECT_PATTERNS.items():
        if pattern.search(text):
            findings.append(f"{label}: {name}")
    if hashed_phrase_hit(text):
        findings.append(f"{label}: hashed identifying phrase")


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, errors="replace")
    return result.stdout + result.stderr


def scan_pdf(path: Path, findings: list[str]) -> None:
    if shutil.which("pdfinfo") is None or shutil.which("pdftotext") is None:
        findings.append(f"{path}: Poppler pdfinfo/pdftotext unavailable")
        return
    info = command_output(["pdfinfo", str(path)])
    text = command_output(["pdftotext", str(path), "-"])
    raw_strings = command_output(["strings", "-a", str(path)]) if shutil.which("strings") else ""
    scan_text(str(path), info + "\n" + text, findings)
    for name, pattern in DIRECT_PATTERNS.items():
        # Compressed PDF streams can contain short printable byte sequences
        # that resemble an email address by chance. Real addresses remain
        # covered by the extracted-text and metadata scan above.
        if name == "email address":
            continue
        if pattern.search(raw_strings):
            findings.append(f"{path}: raw PDF strings contain {name}")
    # Short byte sequences in compressed PDF streams can coincidentally spell a
    # surname or affiliation token. Rendered text and metadata retain the
    # strict one-word check; raw streams require a multi-word identity phrase.
    if hashed_phrase_hit(raw_strings, minimum_word_count=2):
        findings.append(f"{path}: raw PDF strings contain a hashed identifying phrase")
    author_match = re.search(r"(?m)^Author:[ \t]*(.*)$", info)
    if author_match and author_match.group(1).strip():
        findings.append(f"{path}: non-empty PDF Author metadata")


def verify_pdf_contract(path: Path, findings: list[str]) -> None:
    """Require the submitted page count, IEEE letter geometry, and title."""

    expected = PDF_CONTRACT.get(path.name)
    if expected is None or not path.is_file():
        return
    info = command_output(["pdfinfo", str(path)])
    pages_match = re.search(r"(?m)^Pages:\s*(\d+)\s*$", info)
    pages = int(pages_match.group(1)) if pages_match else -1
    if pages != int(expected["pages"]):
        findings.append(f"{path}: expected {expected['pages']} pages, found {pages}")
    size_match = re.search(
        r"(?m)^Page size:\s*([0-9.]+)\s+x\s+([0-9.]+)\s+pts", info
    )
    if size_match is None:
        findings.append(f"{path}: PDF page size unavailable")
    else:
        width, height = map(float, size_match.groups())
        if (
            abs(width - LETTER_WIDTH_PT) > PAGE_TOLERANCE_PT
            or abs(height - LETTER_HEIGHT_PT) > PAGE_TOLERANCE_PT
        ):
            findings.append(f"{path}: expected portrait US letter, found {width:g} x {height:g} pt")
    title_match = re.search(r"(?m)^Title:[ \t]*(.*)$", info)
    title = title_match.group(1).strip() if title_match else ""
    if title != str(expected["title"]):
        findings.append(f"{path}: unexpected PDF title metadata {title!r}")


def scan_binary_payload(path: Path, findings: list[str]) -> None:
    """Scan archived binary payloads, including decompressed NPZ members."""

    payloads: list[tuple[str, bytes]] = []
    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as archive:
                payloads.extend((name, archive.read(name)) for name in archive.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            findings.append(f"{path}: unreadable ZIP/NPZ payload ({exc})")
            return
    else:
        try:
            payloads.append((path.name, path.read_bytes()))
        except OSError as exc:
            findings.append(f"{path}: unreadable binary payload ({exc})")
            return
    for member, payload in payloads:
        # Latin-1 is a one-to-one byte mapping, so ASCII identity strings cannot
        # disappear through decoder error handling.
        scan_text(f"{path}:{member}", payload.decode("latin-1"), findings)


def verify_checksums(root: Path, findings: list[str]) -> None:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        findings.append("SHA256SUMS missing")
        return
    expected = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    observed_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if observed_files != set(expected):
        findings.append("SHA256SUMS file inventory differs from release tree")
        return
    for relative, digest in expected.items():
        if sha256(root / relative) != digest:
            findings.append(f"{relative}: checksum mismatch")


def inspect_tree(root: Path) -> list[str]:
    findings: list[str] = []
    relative_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    missing = REQUIRED_FILES - relative_files
    if missing:
        findings.append(f"missing required files: {sorted(missing)}")
    observed_sections = {
        relative for relative in relative_files if relative.startswith("sections/")
    }
    unexpected_sections = sorted(observed_sections - EXPECTED_SECTIONS)
    if unexpected_sections:
        findings.append(f"unexpected or stale manuscript sections remain: {unexpected_sections}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.casefold().startswith(".git") for part in relative.parts):
            findings.append(f"{relative}: version-control metadata path")
        if not path.is_file():
            continue
        name = path.name.casefold()
        if any(name.endswith(suffix.casefold()) for suffix in BUILD_TRACE_SUFFIXES):
            findings.append(f"{relative}: forbidden build trace")
        if path.suffix.casefold() in TEXT_SUFFIXES or path.name == "SHA256SUMS":
            try:
                scan_text(str(relative), path.read_text(encoding="utf-8"), findings)
            except UnicodeDecodeError:
                findings.append(f"{relative}: expected text is not UTF-8")
        elif path.suffix.casefold() == ".pdf":
            scan_pdf(path, findings)
            verify_pdf_contract(path, findings)
        else:
            scan_binary_payload(path, findings)
    for source in (root / "main.tex", root / "supplement.tex"):
        if source.is_file():
            text = source.read_text(encoding="utf-8")
            if "Anonymous Authors" not in text:
                findings.append(f"{source.name}: anonymous author marker missing")
            if "\\ifdefined" in text or "\\orcidlink" in text or "\\thanks{" in text:
                findings.append(f"{source.name}: conditional/named-author machinery remains")
            if PAPER_TITLE not in text:
                findings.append(f"{source.name}: restored paper title missing")
    main_source = root / "main.tex"
    if main_source.is_file():
        source_text = main_source.read_text(encoding="utf-8")
        missing_inputs = sorted(
            section
            for section in EXPECTED_SECTIONS
            if f"\\input{{{section}}}" not in source_text
        )
        if missing_inputs:
            findings.append(f"main.tex: restored section inputs missing: {missing_inputs}")
    placeholder = root / "anonymous_artifact_url.tex"
    if placeholder.is_file() and "anonymous artifact URL to be supplied" not in placeholder.read_text(
        encoding="utf-8"
    ):
        findings.append("anonymous artifact URL placeholder missing")
    extension_manifest = root / "results/qai_hardware_santa_fe_extension/manifest.json"
    if extension_manifest.is_file():
        try:
            extension = json.loads(extension_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(f"hardware extension manifest is unreadable: {exc}")
        else:
            if extension.get("status") != "COMPLETE_RAW_ARCHIVES_NO_FORECAST_ANALYSIS":
                findings.append("hardware extension manifest does not record a completed archive")
            if extension.get("confirmed_usage_seconds") != 111.0:
                findings.append("hardware extension charged time is not the admitted 111 seconds")
            jobs = extension.get("jobs", {})
            expected_jobs = {
                "ibm_fez": ("d99mf4t2su3c739kl9k0", 55.0),
                "ibm_marrakesh": ("d99mf52f47jc73a99fj0", 56.0),
            }
            for backend, (job_id, usage) in expected_jobs.items():
                job = jobs.get(backend, {})
                if job.get("job_id") != job_id or job.get("usage_seconds") != usage:
                    findings.append(f"{backend}: hardware extension job provenance mismatch")
    verify_checksums(root, findings)
    return findings


def compile_in_temporary_copy(root: Path) -> tuple[dict[str, str], dict[str, int]]:
    if shutil.which("latexmk") is None:
        raise RuntimeError("latexmk is required for the isolated build check")
    env = os.environ.copy()
    env.update(
        {
            "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
            "FORCE_SOURCE_DATE": "1",
            "TZ": "UTC",
        }
    )
    hashes: dict[str, str] = {}
    pages: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="anonymous-release-build-") as temporary:
        build_root = Path(temporary) / "source"
        shutil.copytree(root, build_root)
        for stem in ("main", "supplement"):
            subprocess.run(
                [
                    "latexmk",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"{stem}.tex",
                ],
                cwd=build_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                errors="replace",
            )
            built = build_root / f"{stem}.pdf"
            hashes[f"{stem}.pdf"] = sha256(built)
            info = command_output(["pdfinfo", str(built)])
            match = re.search(r"(?m)^Pages:\s*(\d+)\s*$", info)
            pages[f"{stem}.pdf"] = int(match.group(1)) if match else -1
            findings: list[str] = []
            scan_pdf(built, findings)
            verify_pdf_contract(built, findings)
            if findings:
                raise RuntimeError("isolated built PDF failed scrub: " + "; ".join(findings))
            packaged = root / f"{stem}.pdf"
            if sha256(packaged) != hashes[f"{stem}.pdf"]:
                raise RuntimeError(f"{stem}.pdf is not reproducible from the released source")
    return hashes, pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--build", action="store_true", help="also rebuild both PDFs in a temporary copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    findings = inspect_tree(root)
    if findings:
        for finding in findings:
            print(f"FAIL: {finding}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: source, assets, PDFs, metadata, inventory, and checksums are anonymous-safe ({root})")
    if args.build:
        hashes, pages = compile_in_temporary_copy(root)
        print(f"PASS: isolated reproducible build; pages={pages}; sha256={hashes}")


if __name__ == "__main__":
    main()
