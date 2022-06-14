import torch

from typing import Iterator, Tuple


def class_idx_iterator(one_hot_targets: torch.Tensor) -> Iterator[torch.Tensor]:
    """Get a class-wise iterator over inputs and one-hot targets

    :param one_hot_targets: Batch of targets, one-hot encoded. Must have size (batch_size, num_classes)
    :return: Iterator over (inputs_c, targets_c) for each class c = 0, 1, ..., num_classes-1
    """
    assert len(one_hot_targets.size()) == 2, f"One hot targets are not one-hot! Expected dimension 2 but got shape {one_hot_targets.size()}."
    num_classes = one_hot_targets.size()[-1]

    for cls_idx in range(num_classes):
        yield (torch.argmax(one_hot_targets, dim=-1) == cls_idx).nonzero(as_tuple=True)[0]
