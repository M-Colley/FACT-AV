import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MLP.dataset import encode_trust_value, resolve_trust_class_values


def test_floor_label_mode_keeps_existing_trust_flooring():
    class_values = resolve_trust_class_values(
        pd.Series([1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.0]),
        "floor",
    )

    assert class_values == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert encode_trust_value(1.5, "floor", class_values) == 0
    assert encode_trust_value(3.5, "floor", class_values) == 2
    assert encode_trust_value(4.5, "floor", class_values) == 3


def test_separate_fractional_label_mode_creates_distinct_classes():
    class_values = resolve_trust_class_values(
        pd.Series([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        "separate_fractional",
    )

    assert class_values == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    assert encode_trust_value(1.5, "separate_fractional", class_values) == 1
    assert encode_trust_value(2.5, "separate_fractional", class_values) == 3
    assert encode_trust_value(4.5, "separate_fractional", class_values) == 7
