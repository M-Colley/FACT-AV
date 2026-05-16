"""Smoke check for MLP/eval.py helpers used in publication outputs."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "MLP"))

# Importing the module pulls in plot_calibration and write_per_class_report.
import importlib.util

spec = importlib.util.spec_from_file_location("mlp_eval", REPO / "MLP" / "eval.py")
mlp_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_eval)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 5, size=n)
    probs = rng.dirichlet(np.ones(5) * 0.5, size=n)
    for i in range(n):
        if rng.random() < 0.7:
            probs[i] = np.zeros(5)
            probs[i, y_true[i]] = 0.8
            probs[i, (y_true[i] + 1) % 5] = 0.2

    out_dir = REPO / "results" / "MLP_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # PNG only -- skip PDF here because the matplotlib PDF backend can segfault
    # in headless CI on some Windows / fonttools combinations.
    from plotting_style import apply_paper_style

    apply_paper_style()
    mlp_eval.plot_calibration(
        y_true,
        probs,
        n_bins=10,
        save_path=out_dir / "cal_test",
        title="smoke test",
    )
    mlp_eval.write_per_class_report(
        y_true,
        probs.argmax(axis=1),
        [1.0, 2.0, 3.0, 4.0, 5.0],
        out_dir / "per_class_test.csv",
    )

    artifacts = sorted(p.name for p in out_dir.iterdir())
    print("Created:", artifacts)
    assert (out_dir / "per_class_test.csv").exists()
    print("OK")


if __name__ == "__main__":
    main()
