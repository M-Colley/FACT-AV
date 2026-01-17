from pathlib import Path

import pandas as pd


DATA_FILES = {
    "all_combined_prepared.xlsx": {"mIoU", "Trust1", "SCENARIO", "INTRODUCTION"},
    "all_combined_prepared_removed_REI.xlsx": {"mIoU", "Trust1", "SCENARIO", "INTRODUCTION"},
    "all_combined_prepared_with_demographics.xlsx": {
        "mIoU",
        "Trust1",
        "SCENARIO",
        "INTRODUCTION",
        "Age",
        "Gender",
    },
    "all_combined_prepared_with_demographics_with_baseline.xlsx": {
        "mIoU",
        "Trust1",
        "SCENARIO",
        "INTRODUCTION",
        "Age",
        "Gender",
    },
}


def test_data_files_exist_and_have_required_columns():
    data_dir = Path("data")
    missing_files = [name for name in DATA_FILES if not (data_dir / name).exists()]
    assert not missing_files, f"Missing data files: {missing_files}"

    for name, required_columns in DATA_FILES.items():
        df = pd.read_excel(data_dir / name)
        assert not df.empty, f"{name} is empty"
        missing_columns = required_columns - set(df.columns)
        assert not missing_columns, f"{name} missing columns: {sorted(missing_columns)}"
