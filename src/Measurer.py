import torch
import torch.nn.functional as F
import torch.utils.data
import Models
from DatasetWrapper import DatasetWrapper
import tqdm

import functools
from collections import defaultdict
from typing import Dict, Tuple

import utils

class Measurer:
    def __init__(self):
        pass

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> Dict[str, float]:
        raise NotImplementedError("Measures should overwrite this method!")


class TraceMeasure(Measurer):
    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> Dict[str, float]:
        if shared_cache is None:
            shared_cache = SharedMeasurementVars()

        wrapped_model.base_model.eval()
        data_loader = dataset.train_loader
        measurements = defaultdict(float)
        device = next(wrapped_model.parameters()).device

        for inputs, targets in tqdm.tqdm(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            preds, embeddings = wrapped_model(inputs)  # embeddings: Dict[Hashable, torch.Tensor]
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets

            for class_idx, class_batch_indexes in enumerate(utils.class_idx_iterator(one_hot_targets)):
                for layer_name, activations in embeddings.items():
                    # TODO(marius): Verify calculation (norm over sample, sum over batch)
                    measurements[layer_name+f"/{class_idx}"] += torch.sum(
                        torch.linalg.norm(activations[class_batch_indexes])
                    ).item()

        for layer_name in measurements.keys():
            measurements[layer_name] /= len(data_loader.dataset)

        return measurements


class SharedMeasurementVars:
    """Shared cache of often-used computations used for measurements.

    E.g. class means. This allows them to be calculated once per epoch!
    """
    def __init__(self):
        self._cache = {}
        self._cache_args = {}

    def reset(self):
        """Reset (clean) caches. Should be called every time after epochs (as to not hog memory)."""
        self._cache = {}
        self._cache_args = {}

    def get_train_class_means(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper) -> Dict[str, torch.Tensor]:
        """Get the class means in all embeddings for the train dataset.

        :return: Dict of layer_id to torch tensor of size (num_classes, embedding_shape).
        """
        func_id = self.get_train_class_means
        if func_id in self._cache.keys():
            assert (wrapped_model, dataset) == self._cache_args[func_id], "Got different arguments when calling cache, is this intended use of the cache? Did you reset?"
            return self._cache[func_id]
        self._cache_args[func_id] = (wrapped_model, dataset)

        class_means, class_nums = self._calc_class_means_nums(wrapped_model, dataset.test_loader, dataset.num_classes, dataset.is_one_hot)

        self._cache[self.get_train_class_means] = class_means
        self._cache[self._get_train_class_nums] = class_nums
        self._cache_args[self._get_train_class_nums] = ([dataset], {})

        # Check cache, verifying that the arguments given are the same
        self._cache[func_id] = class_means

        return class_means

    def get_test_class_means(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper) -> Dict[str, torch.Tensor]:
        """Get the class means in all embeddings for the test dataset.

        :return: Dict of layer_id to torch tensor of size (num_classes, embedding_shape).
       """
        func_id = self.get_test_class_means
        if func_id in self._cache.keys():
            assert (wrapped_model, dataset) == self._cache_args[func_id], "Got different arguments when calling cache, is this intended use of the cache? Did you reset?"
            return self._cache[func_id]
        self._cache_args[func_id] = (wrapped_model, dataset)

        class_means, class_nums = self._calc_class_means_nums(wrapped_model, dataset.train_loader, dataset.num_classes, dataset.is_one_hot)

        self._cache[self.get_test_class_means] = class_means
        self._cache[self._get_test_class_nums] = class_nums
        self._cache_args[self._get_test_class_nums] = ([dataset], {})

        return class_means

    @staticmethod
    def _calc_class_means_nums(wrapped_model: Models.ForwardHookedOutput, data_loader: torch.utils.data.DataLoader, num_classes: int, is_one_hot: bool) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        device = next(iter(wrapped_model.parameters())).device
        class_sums = {layer_name: None for layer_name in wrapped_model.output_layers}
        class_nums = torch.zeros((num_classes,))

        i = 0  # TODO(marius): REMOVE DEBUG
        for inputs, targets in tqdm.tqdm(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            preds, embeddings = wrapped_model(inputs)  # embeddings: Dict[Hashable, torch.Tensor]
            one_hot_targets = F.one_hot(targets, num_classes=num_classes) if not is_one_hot else targets

            class_nums += torch.sum(one_hot_targets, dim=0)
            for layer_name, activations in embeddings.items():
                batch_layer_class_means = (
                        one_hot_targets.t().float() @ embeddings[layer_name].detach().transpose(0, -2)
                ).transpose(0, -2).detach()
                if class_sums[layer_name] is None:
                    class_sums[layer_name] = torch.zeros_like(batch_layer_class_means)
                class_sums[layer_name] += batch_layer_class_means

            i += 1
            if i > 1:
                break

        class_means = {}
        for layer_name in wrapped_model.output_layers:
            class_means[layer_name] = (class_sums[layer_name].transpose(0, -1) / class_nums).transpose(0, -1)

        return class_means, class_nums

    def _get_train_class_nums(self, dataset: DatasetWrapper) -> torch.Tensor:
        """Get the number of class samples for each class in the training dataset.

        :return: A tensor [#class_0, #class_1, ...]"""
        class_nums = torch.zeros((dataset.num_classes,))
        for inputs, targets in dataset.train_loader:
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
            class_nums += torch.sum(one_hot_targets, dim=0)

        return class_nums

    def _get_test_class_nums(self, dataset: DatasetWrapper) -> torch.Tensor:
        """Get the number of class samples for each class in the testing dataset.

        :return: A tensor [#class_0, #class_1, ...]"""
        class_nums = torch.zeros((dataset.num_classes,))
        for inputs, targets in dataset.test_loader:
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
            class_nums += torch.sum(one_hot_targets, dim=0)

        return class_nums


def _test_cache():
    from Experiment import Experiment
    config_path = "../config/default.yaml"
    exp = Experiment(config_path)
    # exp.do_measurements()
    cache = SharedMeasurementVars()
    print(cache.get_test_class_means(exp.wrapped_model, exp.dataset)['layer3'].size())
    print(cache.get_test_class_means(exp.wrapped_model, exp.dataset)['layer3'].size())
    cache.reset()
    print(cache.get_test_class_means(exp.wrapped_model, exp.dataset)['layer3'].size())


if __name__ == '__main__':
    _test_cache()


