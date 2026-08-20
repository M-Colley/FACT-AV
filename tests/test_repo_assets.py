"""Documentation-integrity checks: every path the README names must resolve.

This file previously contained three tests that could not fail:

* ``test_readme_assets_exist`` called ``pytest.skip`` on missing files, so a
  deleted figure was reported as a skip rather than a failure;
* ``test_model_json_is_valid`` looked for ``your_model.json`` at the repo root
  while the file was written to ``results/ML-Approaches/`` -- it had therefore
  been permanently skipped, never once validating anything;
* ``test_readme_script_paths_exist`` checked a hard-coded list of exactly one
  script.

The replacement extracts the paths from the README itself, so a renamed script or
a moved output surfaces as a failing test instead of quietly rotting. Paths that
are deliberately untracked (model binaries; see ``.gitignore``) are checked only
when present, since a fresh clone legitimately has not generated them yet.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

# Extensions worth resolving. Anything else in backticks is prose or a CLI flag.
PATH_PATTERN = re.compile(
    r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|png|pdf|json|csv|txt|xlsx|jpg|yml|bat|lock))`"
)

# Generated on demand and deliberately not tracked (large binaries). Validated
# only if a run has produced them.
OPTIONAL_SUFFIXES = (".pt", ".tabpfn_fit")
OPTIONAL_PATHS = {
    "results/ML-Approaches/xgboost_model.json",
}


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def referenced_paths() -> set[str]:
    """Paths from the README that are repo-relative (contain a directory part).

    Bare filenames like ``train.py`` or ``icc.json`` appear in prose and in
    output-listing tables where the directory is given in surrounding text, so
    they are ambiguous and deliberately excluded.
    """
    return {match for match in PATH_PATTERN.findall(readme_text()) if "/" in match}


def test_readme_exists_and_is_not_empty():
    assert README.exists(), "README.md is missing."
    assert len(readme_text()) > 1000


def test_readme_references_at_least_a_few_paths():
    """Guards the regex itself: if it stops matching, the tests below would all
    vacuously pass on an empty set."""
    assert len(referenced_paths()) >= 5


@pytest.mark.parametrize("relative_path", sorted(referenced_paths()))
def test_readme_referenced_path_exists(relative_path):
    path = REPO_ROOT / relative_path
    if relative_path in OPTIONAL_PATHS or path.suffix in OPTIONAL_SUFFIXES:
        if not path.exists():
            pytest.skip(f"{relative_path} is a generated binary — run the pipeline to produce it.")
    assert path.exists(), (
        f"README references {relative_path}, which does not exist. "
        "Either regenerate it or update the README."
    )


@pytest.mark.parametrize(
    "script",
    [
        "ML-approaches.py",
        "cross_validation.py",
        "mixed_effects_baseline.py",
        "publication_figures.py",
        "explainability_extras.py",
        "pysr_config.py",
        "pysr_plots.py",
        "trust_groups.py",
        "main_pysr_trust_calibration.py",
        "main_group_pysr_trust_calibration.py",
        "main_group_pysr_trust_calibration_more_predictors.py",
        "main_personalized_pysr_trust_calibration.py",
        "MLP/train.py",
        "MLP/eval.py",
        "MLP/dataset.py",
        "MLP/network.py",
        "MLP/metrics.py",
        "run_all.py",
    ],
)
def test_entry_point_exists(script):
    assert (REPO_ROOT / script).exists(), f"Missing entry point: {script}"


def test_generated_json_outputs_are_parseable():
    """Every tracked ``.json`` under results/ must be valid JSON.

    A truncated write (interrupted run, full disk) leaves a file that still looks
    present in the tree and still gets committed.
    """
    results = REPO_ROOT / "results"
    if not results.exists():
        pytest.skip("results/ not present — run the analysis scripts first.")

    broken = []
    for path in sorted(results.rglob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            broken.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {exc}")
    assert not broken, "Unparseable JSON under results/: " + "; ".join(broken)


def test_no_stale_test_f1_checkpoints():
    """``test_f1_0.73783.pt`` / ``test_f1_0.74157.pt`` were checkpoints named after
    their *test* score, from a scheme that selected on test rather than validation.
    Their implied ~0.74 F1 contradicts the reported macro-F1 and misleads anyone
    browsing results/. They were deleted; this stops them coming back."""
    stale = sorted(path.name for path in (REPO_ROOT / "results" / "MLP").glob("test_f1_*.pt"))
    assert not stale, (
        f"Checkpoints named after a test score are back: {stale}. "
        "Model selection must use the validation split (see MLP/train.py)."
    )


def _slugify(heading: str) -> str:
    r"""GitHub's heading-anchor rule.

    Lowercase, strip formatting characters, drop everything that is not a word
    character / whitespace / hyphen, then replace **each** whitespace character
    with a hyphen. Runs are *not* collapsed: `## A — B` drops the em dash and
    leaves two spaces, so the anchor is `a--b` with a double hyphen. Collapsing
    them (`\s+`) produces `a-b` and would report every em-dashed heading in this
    README as a broken link.
    """
    text = heading.strip().lower()
    text = re.sub(r"[`*_]", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text).strip("-")


def test_readme_internal_links_resolve():
    """Every ``[text](#anchor)`` must point at a heading that exists.

    The table of contents and the cross-references between sections are the main
    way this README is navigated; a renamed heading silently breaks them and
    nothing else would notice.
    """
    text = readme_text()
    anchors = {_slugify(h) for h in re.findall(r"^#{1,6}\s+(.*)$", text, flags=re.MULTILINE)}
    links = re.findall(r"\]\(#([^)]+)\)", text)

    broken = sorted({link for link in links if link not in anchors})
    assert not broken, (
        f"README links to headings that do not exist: {broken}. "
        "Either the heading was renamed or the link has a typo."
    )
