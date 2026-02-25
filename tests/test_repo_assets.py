import json
from pathlib import Path


ASSET_PATHS = [
    Path("results/ML-Approaches/feature_importance_random_classifier.png"),
    Path("results/PySR/relationship_pysr_other_rows_df_all_combined_prepared_legend.png"),
    Path("MLP/epochs/epoch1990.jpg"),
]

README_SCRIPT_PATHS = [
    Path("ML-approaches.py"),
]


def test_readme_assets_exist():
    missing = [path.as_posix() for path in ASSET_PATHS if not path.exists()]
    assert not missing, f"Missing README assets: {missing}"


def test_model_json_is_valid():
    model_path = Path("your_model.json")
    assert model_path.exists(), "your_model.json is missing"
    with model_path.open("r", encoding="utf-8") as handle:
        json.load(handle)


def test_readme_script_paths_exist():
    missing = [path.as_posix() for path in README_SCRIPT_PATHS if not path.exists()]
    assert not missing, f"Missing README script paths: {missing}"
