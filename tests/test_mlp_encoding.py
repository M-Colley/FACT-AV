"""Tests for MLP encoding helpers in MLP/dataset.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MLP.dataset import (
    TRUST_LABEL_MODES,
    encode_distance,
    encode_driving,
    encode_education,
    encode_gender,
    encode_intro,
    encode_job,
    encode_license,
    encode_scenario,
    encode_trust_value,
    resolve_trust_class_values,
)


# ---------------------------------------------------------------------------
# encode_scenario
# ---------------------------------------------------------------------------

class TestEncodeScenario:
    VALID_SCENARIOS = ["3Spurig", "Spielstrasse", "Ueberland", "NeueMitte"]

    def test_all_valid_scenarios_produce_4d_one_hot(self):
        for scenario in self.VALID_SCENARIOS:
            result = encode_scenario(scenario)
            assert result.shape == (4,), f"Wrong shape for {scenario}"
            assert result.sum() == 1.0, f"Not one-hot for {scenario}"

    def test_3spurig_is_first_class(self):
        result = encode_scenario("3Spurig")
        assert result[0] == 1.0
        assert result[1:].sum() == 0.0

    def test_neue_mitte_is_last_class(self):
        result = encode_scenario("NeueMitte")
        assert result[3] == 1.0
        assert result[:3].sum() == 0.0

    def test_invalid_scenario_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown value for SCENARIO"):
            encode_scenario("Highway")

    def test_output_is_float32(self):
        result = encode_scenario("Spielstrasse")
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# encode_intro
# ---------------------------------------------------------------------------

class TestEncodeIntro:
    def test_ambiguous_returns_zero(self):
        assert encode_intro("ambiguous").item() == 0.0

    def test_ambigious_typo_accepted_returns_zero(self):
        assert encode_intro("ambigious").item() == 0.0

    def test_boasting_returns_one(self):
        assert encode_intro("boasting").item() == 1.0

    def test_case_insensitive_ambiguous(self):
        assert encode_intro("AMBIGUOUS").item() == 0.0

    def test_case_insensitive_boasting(self):
        assert encode_intro("BOASTING").item() == 1.0

    def test_leading_trailing_whitespace_stripped(self):
        assert encode_intro("  boasting  ").item() == 1.0

    def test_invalid_intro_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown value for INTRODUCTION"):
            encode_intro("neutral")

    def test_output_is_float32(self):
        result = encode_intro("boasting")
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# encode_gender / encode_education / encode_job / encode_driving / encode_distance
# ---------------------------------------------------------------------------

class TestEncodeGender:
    def test_valid_values_produce_4d_one_hot(self):
        for code in ["A1", "A2", "A3", "A4"]:
            result = encode_gender(code)
            assert result.shape == (4,)
            assert result.sum() == 1.0

    def test_a1_is_first_class(self):
        result = encode_gender("A1")
        assert result[0] == 1.0

    def test_a5_is_invalid_for_gender(self):
        with pytest.raises(ValueError):
            encode_gender("A5")


class TestEncodeEducation:
    def test_valid_values_produce_5d_one_hot(self):
        for code in ["A1", "A2", "A3", "A4", "A5"]:
            result = encode_education(code)
            assert result.shape == (5,)
            assert result.sum() == 1.0

    def test_a6_is_invalid_for_education(self):
        with pytest.raises(ValueError):
            encode_education("A6")


class TestEncodeJob:
    def test_valid_values_produce_6d_one_hot(self):
        for code in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            result = encode_job(code)
            assert result.shape == (6,)
            assert result.sum() == 1.0


class TestEncodeDriving:
    def test_valid_values_produce_6d_one_hot(self):
        for code in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            result = encode_driving(code)
            assert result.shape == (6,)
            assert result.sum() == 1.0


class TestEncodeDistance:
    def test_valid_values_produce_5d_one_hot(self):
        for code in ["A1", "A2", "A3", "A4", "A5"]:
            result = encode_distance(code)
            assert result.shape == (5,)
            assert result.sum() == 1.0

    def test_a6_is_invalid_for_distance(self):
        with pytest.raises(ValueError):
            encode_distance("A6")


# ---------------------------------------------------------------------------
# encode_license
# ---------------------------------------------------------------------------

class TestEncodeLicense:
    def test_y_returns_one(self):
        result = encode_license("Y")
        assert result.item() == 1.0

    def test_n_returns_zero(self):
        result = encode_license("N")
        assert result.item() == 0.0

    def test_any_non_y_returns_zero(self):
        assert encode_license("N").item() == 0.0
        assert encode_license("no").item() == 0.0
        assert encode_license("").item() == 0.0

    def test_output_shape(self):
        assert encode_license("Y").shape == (1,)


# ---------------------------------------------------------------------------
# resolve_trust_class_values
# ---------------------------------------------------------------------------

class TestResolveTrustClassValues:
    def test_floor_mode_returns_five_integer_classes(self):
        series = pd.Series([1.0, 1.5, 2.5, 3.5, 4.5, 5.0])
        result = resolve_trust_class_values(series, "floor")
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_floor_mode_ignores_half_steps(self):
        series = pd.Series([1.5, 3.5])
        result = resolve_trust_class_values(series, "floor")
        assert len(result) == 5

    def test_separate_fractional_keeps_observed_values(self):
        series = pd.Series([1.0, 1.5, 2.0, 2.5, 3.0])
        result = resolve_trust_class_values(series, "separate_fractional")
        assert result == [1.0, 1.5, 2.0, 2.5, 3.0]

    def test_separate_fractional_deduplicates(self):
        series = pd.Series([1.0, 1.0, 2.0, 2.0])
        result = resolve_trust_class_values(series, "separate_fractional")
        assert result == [1.0, 2.0]

    def test_separate_fractional_sorted(self):
        series = pd.Series([3.0, 1.0, 2.0])
        result = resolve_trust_class_values(series, "separate_fractional")
        assert result == sorted(result)

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown trust_label_mode"):
            resolve_trust_class_values(pd.Series([1.0, 2.0]), "invalid")


# ---------------------------------------------------------------------------
# encode_trust_value
# ---------------------------------------------------------------------------

class TestEncodeTrustValue:
    FLOOR_CLASSES = [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_floor_exact_integer_3(self):
        assert encode_trust_value(3.0, "floor", self.FLOOR_CLASSES) == 2

    def test_floor_fractional_floors_down(self):
        assert encode_trust_value(3.5, "floor", self.FLOOR_CLASSES) == 2

    def test_floor_1_5_maps_to_class_0(self):
        assert encode_trust_value(1.5, "floor", self.FLOOR_CLASSES) == 0

    def test_floor_4_5_maps_to_class_3(self):
        assert encode_trust_value(4.5, "floor", self.FLOOR_CLASSES) == 3

    def test_floor_clips_minimum_at_1(self):
        assert encode_trust_value(1.0, "floor", self.FLOOR_CLASSES) == 0

    def test_floor_clips_maximum_at_5(self):
        assert encode_trust_value(5.0, "floor", self.FLOOR_CLASSES) == 4

    def test_separate_fractional_exact_match(self):
        classes = [1.0, 1.5, 2.0, 2.5, 3.0]
        assert encode_trust_value(2.5, "separate_fractional", classes) == 3

    def test_separate_fractional_first_class(self):
        classes = [1.0, 2.0, 3.0]
        assert encode_trust_value(1.0, "separate_fractional", classes) == 0

    def test_separate_fractional_unknown_value_raises(self):
        classes = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            encode_trust_value(1.5, "separate_fractional", classes)

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown trust_label_mode"):
            encode_trust_value(3.0, "bad_mode", self.FLOOR_CLASSES)


# ---------------------------------------------------------------------------
# TRUST_LABEL_MODES constant
# ---------------------------------------------------------------------------

class TestTrustLabelModes:
    def test_floor_is_present(self):
        assert "floor" in TRUST_LABEL_MODES

    def test_separate_fractional_is_present(self):
        assert "separate_fractional" in TRUST_LABEL_MODES

    def test_modes_is_tuple_or_sequence(self):
        assert len(TRUST_LABEL_MODES) >= 2
