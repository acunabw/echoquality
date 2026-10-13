"""Regression tests for minority-class-safe, patient-disjoint CV."""
from __future__ import annotations

import numpy as np
import pytest

from echoquality.recognition.models import validated_group_folds


def test_validated_group_folds_keep_both_classes_and_disjoint_patients() -> None:
    # Only two minority-positive patients among 12; each has two views.
    groups = np.repeat([f"patient_{i:02d}" for i in range(12)], 2)
    y = np.zeros(24, dtype=int)
    y[0] = 1
    y[8] = 1
    folds = validated_group_folds(y, groups, 2, seed=2027)
    assert len(folds) == 2
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(len(y)))
    for train, test in folds:
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert set(y[train]) == {0, 1}
        assert set(y[test]) == {0, 1}


def test_single_minority_patient_fails_explicitly() -> None:
    groups = np.repeat([f"p{i}" for i in range(10)], 2)
    y = np.zeros(20, dtype=int)
    y[0] = 1
    with pytest.raises(ValueError, match="two distinct patient groups"):
        validated_group_folds(y, groups, 2, seed=2027)


def test_single_class_fails_explicitly() -> None:
    groups = np.repeat([f"p{i}" for i in range(10)], 2)
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="both classes"):
        validated_group_folds(y, groups, 2, seed=2027)
