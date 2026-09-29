import numpy as np
import pytest
import torch

from gesturevision.digit_data import stratified_group_split
from gesturevision.digit_models import SUPPORTED_IMAGE_MODELS, build_digit_model


def synthetic_labels_and_duplicate_groups():
    labels = np.repeat(np.arange(10), 40)
    groups = []
    for digit in range(10):
        for group_id in range(20):
            groups.extend([f"digit-{digit}-source-{group_id}"] * 2)
    return labels, np.asarray(groups)


def test_stratified_group_split_is_repeatable_and_leakage_free():
    labels, groups = synthetic_labels_and_duplicate_groups()
    first = stratified_group_split(labels, groups, seed=19)
    second = stratified_group_split(labels, groups, seed=19)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    expected = set(np.unique(labels))
    group_sets = [set(groups[index]) for index in first]
    for indices in first:
        assert set(np.unique(labels[indices])) == expected
    assert not (group_sets[0] & group_sets[1])
    assert not (group_sets[0] & group_sets[2])
    assert not (group_sets[1] & group_sets[2])
    assert len(first[0]) > len(first[1]) > 0
    assert len(first[2]) > 0


def test_stratified_group_split_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="equal length"):
        stratified_group_split(np.arange(50), np.arange(49))


@pytest.mark.parametrize("model_name", SUPPORTED_IMAGE_MODELS)
def test_digit_backbones_return_ten_logits(model_name):
    model = build_digit_model(model_name, num_classes=10, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 160, 160))
    assert output.shape == (1, 10)
    assert torch.isfinite(output).all()
