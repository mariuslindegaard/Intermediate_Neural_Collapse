import torch
import torch.nn.functional as F
import torch.utils.data
import Models
from DatasetWrapper import DatasetWrapper
import pandas as pd

from collections import defaultdict
from typing import Dict, Tuple, List, Any, Hashable

import utils


class Measurer:
    def __init__(self):
        pass

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        """Measurement method taking the model and dataset to generate measurements.

        Output is a list with one entry per measurement. "value" is the value of the data, with all other dict entries seen as data parameters.
        """
        raise NotImplementedError("Measures should overwrite this method!")


class AccuracyMeasure(Measurer):
    """Measure test and train accuracy"""
    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVars()
        wrapped_model.eval()  # TODO(marius): Might throw error. If not: Change in TraceMeasure too
        device = next(wrapped_model.parameters()).device

        dataset_splits = {
            'train': dataset.train_loader,
            'test': dataset.test_loader,
        }


        out: List[Dict[str, Any]] = []  # Output to return. List of different value entries, one for each datapoint.
        for split_id, data_loader in dataset_splits.items():
            num_samples, correct = 0, 0
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds, embeddings = wrapped_model(inputs)
                # one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
                class_idx_targets = torch.argmax(targets, dim=-1) if dataset.is_one_hot else targets

                correct += torch.argmax(preds, dim=-1).eq(class_idx_targets).sum().item()
                num_samples += len(inputs)

            accuracy = correct / num_samples
            out.append({'value': accuracy, 'split': split_id})

        return pd.DataFrame(out)


class TraceMeasure(Measurer):
    """Measure traces of activations in relevant layers"""
    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVars()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device
        data_loader = dataset.train_loader

        class_trace_sums: Dict[Hashable, Dict[int, float]] = defaultdict(lambda: defaultdict(float))  # Dict of, for each layer a dict of class number to total norm.
        class_num_samples = torch.zeros(dataset.num_classes, device=device)

        class_means = shared_cache.get_test_class_means(wrapped_model, dataset)

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds, embeddings = wrapped_model(inputs)  # embeddings: Dict[Hashable, torch.Tensor]
            # print(torch.mean(torch.eq(preds, targets)))
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets

            for class_idx, class_batch_indexes in enumerate(utils.class_idx_iterator(one_hot_targets)):
                class_num_samples[class_idx] += len(class_batch_indexes)
                for layer_name, activations in embeddings.items():
                    # TODO(marius): Verify calculation (norm over sample, sum over batch)
                    class_trace_sums[layer_name][class_idx] += torch.sum(
                        torch.linalg.norm(activations[class_batch_indexes] - class_means[layer_name][class_idx]) ** 2
                    ).item()

        global_mean = {}
        for layer_name, layer_class_mean in class_means.items():
            global_mean[layer_name] = (layer_class_mean.transpose(0, -1) @ class_num_samples.view(-1, 1)
                                          / torch.sum(class_num_samples)).transpose(0, -1)

        out: List[Dict[str, Any]] = []  # Output to return. List of different value entries, one for each datapoint.
        for layer_name, layer_dict in class_trace_sums.items():
            within_class_trc = 0
            between_class_trc = 0
            for class_idx in layer_dict.keys():
                layer_class_trace_mean = class_trace_sums[layer_name][class_idx] / class_num_samples[class_idx]
                # out.append({'value': layer_class_average, 'layer_name': layer_name, 'class': class_idx})
                within_class_trc += (layer_class_trace_mean / dataset.num_classes).item()
                between_class_trc += (
                        torch.linalg.norm(class_means[layer_name][class_idx] - global_mean[layer_name]) ** 2
                        / dataset.num_classes
                ).item()
            out.append({'value': within_class_trc, 'layer_name': layer_name, 'trace': 'within'})
            out.append({'value': between_class_trc, 'layer_name': layer_name, 'trace': 'between'})
            out.append({'value': within_class_trc + between_class_trc, 'layer_name': layer_name, 'trace': 'sum'})

        return pd.DataFrame(out)


class SharedMeasurementVars:
    """Shared cache of often-used computations used for measurements.

    E.g. class means. This allows them to be calculated only once per epoch!
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
        class_trace_sums = {layer_name: None for layer_name in wrapped_model.output_layers}
        class_nums = torch.zeros((num_classes,)).to(device)

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds, embeddings = wrapped_model(inputs)  # embeddings: Dict[Hashable, torch.Tensor]
            one_hot_targets = F.one_hot(targets, num_classes=num_classes) if not is_one_hot else targets

            class_nums += torch.sum(one_hot_targets, dim=0)
            for layer_name, activations in embeddings.items():
                batch_layer_class_means = (
                        one_hot_targets.t().float() @ activations.detach().transpose(0, -2)
                ).transpose(0, -2).detach()
                if class_trace_sums[layer_name] is None:
                    class_trace_sums[layer_name] = torch.zeros_like(batch_layer_class_means)
                class_trace_sums[layer_name] += batch_layer_class_means

        class_means = {}
        for layer_name in wrapped_model.output_layers:
            class_means[layer_name] = (class_trace_sums[layer_name].transpose(0, -1) / class_nums).transpose(0, -1)

        return class_means, class_nums

    @staticmethod
    def _get_train_class_nums(dataset: DatasetWrapper) -> torch.Tensor:
        """Get the number of class samples for each class in the training dataset.

        :return: A tensor [#class_0, #class_1, ...]"""
        class_nums = torch.zeros((dataset.num_classes,))
        for inputs, targets in dataset.train_loader:
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
            class_nums += torch.sum(one_hot_targets, dim=0)

        return class_nums

    @staticmethod
    def _get_test_class_nums(dataset: DatasetWrapper) -> torch.Tensor:
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
