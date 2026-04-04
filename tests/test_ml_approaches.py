"""Tests for ML-approaches.py helpers (Config, DataProcessor, utility functions)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ML-approaches.py uses a hyphen so it cannot be imported with a plain `import`.
_spec = importlib.util.spec_from_file_location(
    "ml_approaches",
    Path(__file__).resolve().parents[1] / "ML-approaches.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

Config = _module.Config
DataProcessor = _module.DataProcessor
prepare_categorical_as_string = _module.prepare_categorical_as_string
get_tabpfn_quantile_columns = _module.get_tabpfn_quantile_columns


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_target_column(self):
        config = Config()
        assert config.target_column == "trust"

    def test_default_test_size(self):
        config = Config()
        assert config.test_size == 0.2

    def test_default_random_state(self):
        config = Config()
        assert config.random_state == 42

    def test_default_numerical_features_contains_miou(self):
        config = Config()
        assert "mIoU" in config.numerical_features

    def test_default_categorical_features_contains_scenario(self):
        config = Config()
        assert "SCENARIO" in config.categorical_features

    def test_default_categorical_features_contains_introduction(self):
        config = Config()
        assert "INTRODUCTION" in config.categorical_features

    def test_results_path_is_created_on_init(self, tmp_path):
        config = Config(results_path=tmp_path / "test_results")
        assert config.results_path.exists()

    def test_bootstrap_n_default(self):
        config = Config()
        assert config.bootstrap_n == 20


# ---------------------------------------------------------------------------
# DataProcessor label mappings
# ---------------------------------------------------------------------------

class TestDataProcessorLabelMappings:
    @pytest.fixture
    def processor(self):
        return DataProcessor(Config())

    def test_has_gender_mapping(self, processor):
        assert "Gender" in processor.get_label_mappings()

    def test_gender_a1_is_female(self, processor):
        assert processor.get_label_mappings()["Gender"]["A1"] == "F"

    def test_gender_a2_is_male(self, processor):
        assert processor.get_label_mappings()["Gender"]["A2"] == "M"

    def test_driving_frequency_has_six_values(self, processor):
        freq = processor.get_label_mappings()["DrivingFrequency"]
        assert len(freq) == 6

    def test_education_has_five_values(self, processor):
        edu = processor.get_label_mappings()["Education"]
        assert len(edu) == 5

    def test_job_has_six_values(self, processor):
        job = processor.get_label_mappings()["Job"]
        assert len(job) == 6

    def test_distance_has_five_values(self, processor):
        dist = processor.get_label_mappings()["Distance"]
        assert len(dist) == 5


class TestDataProcessorFeatureNameMappings:
    @pytest.fixture
    def processor(self):
        return DataProcessor(Config())

    def test_scenario_neue_mitte_maps_to_city(self, processor):
        mappings = processor.get_feature_name_mappings()
        assert mappings.get("SCENARIO_NeueMitte") == "Scenario: City"

    def test_scenario_ueberland_maps_to_cross_country(self, processor):
        mappings = processor.get_feature_name_mappings()
        assert mappings.get("SCENARIO_Ueberland") == "Scenario: Cross-Country"

    def test_license_has_friendly_name(self, processor):
        mappings = processor.get_feature_name_mappings()
        assert "License" in mappings

    def test_miou_not_in_mappings(self, processor):
        # mIoU is not renamed — it keeps its original label
        mappings = processor.get_feature_name_mappings()
        assert "mIoU" not in mappings


class TestDataProcessorLoadErrors:
    def test_missing_file_raises_file_not_found(self):
        config = Config(data_path=Path("nonexistent_file.xlsx"))
        processor = DataProcessor(config)
        with pytest.raises(FileNotFoundError):
            processor.load_and_preprocess_data()


# ---------------------------------------------------------------------------
# prepare_categorical_as_string
# ---------------------------------------------------------------------------

class TestPrepareCategoricalAsString:
    def test_converts_specified_columns_to_string_dtype(self):
        df = pd.DataFrame({"cat": [1, 2, 3], "num": [4.0, 5.0, 6.0]})
        result = prepare_categorical_as_string(df, ["cat"])
        assert pd.api.types.is_string_dtype(result["cat"])

    def test_leaves_unspecified_columns_unchanged(self):
        df = pd.DataFrame({"cat": [1, 2, 3], "num": [4.0, 5.0, 6.0]})
        result = prepare_categorical_as_string(df, ["cat"])
        assert result["num"].dtype == float

    def test_does_not_modify_original_dataframe(self):
        df = pd.DataFrame({"cat": [1, 2, 3]})
        original_dtype = df["cat"].dtype
        _ = prepare_categorical_as_string(df, ["cat"])
        assert df["cat"].dtype == original_dtype

    def test_converts_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5.0, 6.0]})
        result = prepare_categorical_as_string(df, ["a", "b"])
        assert pd.api.types.is_string_dtype(result["a"])
        assert pd.api.types.is_string_dtype(result["b"])
        assert result["c"].dtype == float


# ---------------------------------------------------------------------------
# get_tabpfn_quantile_columns
# ---------------------------------------------------------------------------

class TestGetTabpfnQuantileColumns:
    def test_1d_array_single_quantile(self):
        arr = np.array([0.1, 0.5, 0.9])
        result = get_tabpfn_quantile_columns(arr, [0.5])
        assert len(result) == 1
        q, preds = result[0]
        assert q == 0.5
        np.testing.assert_array_equal(preds, arr)

    def test_2d_columns_orientation(self):
        # shape (n_samples, n_quantiles)
        arr = np.zeros((5, 3))
        arr[:, 0] = 0.1
        arr[:, 1] = 0.5
        arr[:, 2] = 0.9
        quantiles = [0.25, 0.5, 0.75]
        result = get_tabpfn_quantile_columns(arr, quantiles)
        assert len(result) == 3
        assert result[0][0] == 0.25
        assert result[1][0] == 0.5
        assert result[2][0] == 0.75

    def test_2d_rows_orientation(self):
        # shape (n_quantiles, n_samples)
        arr = np.zeros((3, 5))
        quantiles = [0.25, 0.5, 0.75]
        result = get_tabpfn_quantile_columns(arr, quantiles)
        assert len(result) == 3

    def test_1d_with_multiple_quantiles_raises(self):
        arr = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError):
            get_tabpfn_quantile_columns(arr, [0.25, 0.5, 0.75])

    def test_ambiguous_2d_shape_raises(self):
        # 4×4 with 3 quantiles — neither dim matches
        arr = np.zeros((4, 4))
        with pytest.raises(ValueError):
            get_tabpfn_quantile_columns(arr, [0.25, 0.5, 0.75])

    def test_3d_array_raises(self):
        arr = np.zeros((2, 3, 4))
        with pytest.raises(ValueError):
            get_tabpfn_quantile_columns(arr, [0.5])

    def test_predictions_are_numpy_arrays(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = get_tabpfn_quantile_columns(arr, [0.5])
        assert isinstance(result[0][1], np.ndarray)
