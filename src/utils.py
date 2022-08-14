import torch
import pandas as pd

from typing import Iterator, Tuple, Union, Iterable, Hashable, List
import functools


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
                             require_leaf: bool = None) -> Iterator[Tuple[str, torch.nn.Module]]:
    """Get all named modules of the model fitting the layer_names (or all layers if layer_names==True).

    :param model: Model to get layers of
    :param layer_names: Name of layers (equivalent to 'model.blockname.layername
    :param require_leaf: Whether to only allow leafs (modules with no children). If not given defaults to 'False' unless layer_names is True.
    :return: Iterator over (module_name, module) tuples
    """
    yield_all = layer_names is True
    if require_leaf is None:
        require_leaf = yield_all

    # TODO(marius): Include check for if layers specifications are never contained in model
    for name, module, in model.named_modules():
        # Don't include if the module has children and we only want leaf modules.
        if require_leaf and len(module.children()) != 0:
            continue

        if yield_all:
            yield name, module
        elif sum(map(  # Check if the condition holds for one of the name specifications:
                lambda key: ("^" + name).endswith(str(key)),
                layer_names
        )):
            yield name, module


def slice_to_smaller_batch(*args: Tuple[torch.Tensor, ...], batch_size: int = 1) -> Iterator[Tuple[torch.Tensor, ...]]:
    """Yields the input arguments sliced into in smaller batches.

    Typically meant to be used in:
    for sliced_inputs, sliced_targets in slice_to_smaller_batch(batch_inputs, batch_targets, batch_size=16):
        ...


    :param args: Tensors to slice
    :param batch_size: Maximal batch size to use.
    :return: Iterator over the sliced input tensors.
    """
    total_samples = len(args[0])

    for first_sample_idx in range(0, len(args[0]), batch_size):
        ret = tuple(
            batch_tensor[first_sample_idx:min(first_sample_idx+batch_size, total_samples)]
            for batch_tensor in args
        )
        yield ret


# From: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    """Implementation of recursive getattr for nested objects"""
    def _getattr(obj, attr):
        try:
            integer_index = int(attr)
            return obj[integer_index]
        except ValueError as e:
            return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def corr_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get a correlation dataframe (as in pd.DF.corr()) from a dataframe with (value, l_type, l_ord, r_type, r_ord)

    :param df: Dataframe with (value, l_type, l_ord, r_type, r_ord) as columns
    :return: Correlation matrix with left-values as row-indices and right-values as columns.
    """
    columns = []
    typename_map = lambda typename: ("w" if str(typename) == '-1' else f"c{typename}")
    for r_type in df['r_type'].unique():
        r_type_selection = df['r_type'] == r_type
        for r_ord in df[r_type_selection]['r_ord'].unique():
            # Get the selection of data for this right-side-specification (r_type, r_ord)
            r_selection = r_type_selection & (df[r_type_selection]['r_ord'] == r_ord)
            # Define column and row names by i.e. the same operation
            col_name = typename_map(r_type) + f"_{r_ord}"
            row_names = df[r_selection]['l_type'].map(typename_map) + '_' + df[r_selection]['l_ord'].map(str)

            # Set the data and store the series
            column = pd.Series(df[r_selection]['value'].to_numpy(), index=row_names, name=col_name)
            columns.append(column)

    ret = pd.DataFrame(columns).T
    return ret
