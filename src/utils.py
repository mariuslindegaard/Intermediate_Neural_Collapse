import torch

from typing import Iterator, Tuple, Union, Iterable, Hashable


def class_idx_iterator(one_hot_targets: torch.Tensor) -> Iterator[torch.Tensor]:
    """Get a class-wise iterator over inputs and one-hot targets

    :param one_hot_targets: Batch of targets, one-hot encoded. Must have size (batch_size, num_classes)
    :return: Iterator over (inputs_c, targets_c) for each class c = 0, 1, ..., num_classes-1
    """
    assert len(one_hot_targets.size()) == 2, f"One hot targets are not one-hot! Expected dimension 2 but got shape {one_hot_targets.size()}."
    num_classes = one_hot_targets.size()[-1]

    for cls_idx in range(num_classes):
        yield (torch.argmax(one_hot_targets, dim=-1) == cls_idx).nonzero(as_tuple=True)[0]


def filter_all_named_modules(model: torch.nn.Module, layer_names: Union[Iterable[Hashable], bool],
                             require_leaf: bool = False) -> Iterator[Tuple[str, torch.nn.Module]]:
    """Get all named modules of the model fitting the layer_names (or all layers if layer_names==True).

    :param model: Model to get layers of
    :param layer_names: Name of layers (equivalent to 'model.blockname.layername
    :param require_leaf: Whether to only allow leafs (modules with no children).
    :return: Iterator over (module_name, module) tuples
    """
    yield_all = layer_names is True

    for name, module, in model.named_modules():
        # Don't include if the module has children and we only want leaf modules.
        if require_leaf and len(module.children()) != 0:
            continue

        if yield_all:
            yield name, module
        elif sum(map(lambda key: name.endswith(str(key)), layer_names)):
            yield name, module

