import torch
import torch.nn.functional as F
import torch.utils.data
import Models
from DatasetWrapper import DatasetWrapper
import pandas as pd
import numpy as np
import scipy.sparse.linalg

from collections import defaultdict, OrderedDict
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

        class_trace_sums: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))  # Dict of, for each layer a dict of class number to total norm.
        # class_num_samples = torch.zeros(dataset.num_classes, device=device)

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds, embeddings = wrapped_model(inputs)  # embeddings: Dict[Hashable, torch.Tensor]
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets

            for class_idx, class_batch_indexes in enumerate(utils.class_idx_iterator(one_hot_targets)):
                # class_num_samples[class_idx] += len(class_batch_indexes)
                for layer_name, activations in embeddings.items():
                    # TODO(marius): Verify calculation (norm over sample, sum over batch)
                    class_trace_sums[layer_name][class_idx] += torch.sum(
                        torch.linalg.norm(activations[class_batch_indexes] - class_means[layer_name][class_idx]) ** 2
                    ).item()

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


class CDNVMeasure(Measurer):
    """Measure the CDNV metric introduced for neural collapse by Tomer Galanti."""
    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVars()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)

        total_activation_var: Dict[str, torch.Tensor] = OrderedDict()  # \Sigma_w
        # NCC_match_net = 0

        # M = torch.stack(mean).T  # Mean of classes before layer

        for inputs, targets in dataset.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            embeddings: Dict[str, torch.Tensor]
            preds, embeddings = wrapped_model(inputs)
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
            # class_idx_targets = torch.argmax(targets, dim=-1) if dataset.is_one_hot else targets

            for layer_name, activations in embeddings.items():
                # flat_activations = torch.flatten(activations, start_dim=1)
                if layer_name not in total_activation_var.keys():
                    total_activation_var[layer_name] = torch.zeros(dataset.num_classes)

                for class_idx, class_batch_indexes in enumerate(utils.class_idx_iterator(one_hot_targets)):
                    if not len(class_batch_indexes):  # Continue if no images classified to this class
                        continue
                    class_activations = activations[class_batch_indexes, :].detach()

                    # update within-class cov
                    rel_class_activations = class_activations - class_means[layer_name][class_idx]
                    total_activation_var[layer_name] += torch.sum(rel_class_activations ** 2).detach().cpu()

        out: List[Dict[str, Any]] = []
        for layer_name, total_var in total_activation_var.items():

            var_per_sample = total_var.cpu() / class_num_samples.cpu()
            rel_class_means = torch.flatten(class_means[layer_name] - global_mean[layer_name], start_dim=1).cpu()

            mean_diffs = torch.cdist(rel_class_means, rel_class_means)
            var_sums = var_per_sample.reshape(-1, 1) + var_per_sample
            cdnv = var_sums / torch.clamp(2*mean_diffs, min=1E-15)  # Clamp to avoid div by 0
            cdnv.fill_diagonal_(0)
            sum_cdnv = torch.sum(cdnv).item()

            out.append({'value': sum_cdnv, 'layer_name': layer_name})

        return pd.DataFrame(out)


class NC1Measure(Measurer):
    """Measure NC metrics"""
    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVars()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)

        cov_within: Dict[str, torch.Tensor] = OrderedDict()  # \Sigma_w
        # NCC_match_net = 0

        # M = torch.stack(mean).T  # Mean of classes before layer

        for inputs, targets in dataset.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            embeddings: Dict[str, torch.Tensor]
            preds, embeddings = wrapped_model(inputs)
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
            # class_idx_targets = torch.argmax(targets, dim=-1) if dataset.is_one_hot else targets

            for layer_name, activations in embeddings.items():
                flat_activations = torch.flatten(activations, start_dim=1)
                if layer_name not in cov_within.keys():
                    activation_size = flat_activations.size()[-1]
                    cov_within[layer_name] = torch.zeros((activation_size, activation_size), device='cpu')

                for class_idx, class_batch_indexes in enumerate(utils.class_idx_iterator(one_hot_targets)):
                    if not len(class_batch_indexes):  # Continue if no images classified to this class
                        continue
                    class_activations = flat_activations[class_batch_indexes, :].detach()

                    # update within-class cov
                    rel_class_activations = class_activations - class_means[layer_name][class_idx].reshape(1, -1)
                    cov = torch.matmul(rel_class_activations.unsqueeze(-1),  # B CHW 1
                                       rel_class_activations.unsqueeze(1))  # B 1 CHW
                    cov_within[layer_name] += torch.sum(cov, dim=0).detach().to('cpu')

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    # net_pred = torch.argmax(outputs[idxs, :], dim=1).cpu()
                    # true_class = torch.argmax(labels[idxs, :], dim=1).cpu()
                    # net_correct += sum(net_pred == true_class).item()

                    # 2) agreement between prediction and nearest class center
                    # NCC_scores = torch.stack([torch.norm(h_c[i, :] - M.T, dim=1) \
                    #                           for i in range(h_c.shape[0])])
                    # NCC_pred = torch.argmin(NCC_scores, dim=1).cpu()
                    # NCC_match_net += sum(NCC_pred == net_pred).item()
                    print(f"Class: {class_idx}")

        # Make cov_within an average instead of a sum
        for layer_name, cov_within_sum in cov_within.items():
            cov_within[layer_name] = cov_within_sum / torch.sum(class_num_samples)

        # Calculate NC1-condition and add to output
        out: List[Dict[str, Any]] = []
        for layer_name in global_mean.keys():
            rel_class_means = class_means[layer_name] - global_mean[layer_name]
            layer_cov_within = cov_within[layer_name]

            layer_cov_between = torch.matmul(rel_class_means, rel_class_means.T) / dataset.num_classes  # TODO(marius): Verify calculation
            S_within = layer_cov_within.cpu().numpy()
            S_between = layer_cov_between.cpu().numpy()
            eigvecs, eigvals, _ = scipy.sparse.linalg.svds(S_between, k=dataset.num_classes-1)
            inv_S_between = eigvecs @ np.diag(eigvals ** (-1)) @ eigvecs.T  # Will get divide by 0 for first epochs, it is fine
            nc1_value = np.trace(S_within @ inv_S_between)  # \Sigma_w @ \Sigma_b^-1
            out.append({'value': nc1_value, 'layer_name': layer_name})


class SharedMeasurementVars:
    """Shared cache of often-used computations used for measurements.

    E.g. class means. This allows them to be calculated only once per epoch!
    """
    def __init__(self):
        self._cache: Dict[callable, Tuple[Dict[str, torch.Tensor], torch.Tensor]] = {}
        self._cache_args: Dict[callable, Tuple[Models.ForwardHookedOutput, DatasetWrapper]] = {}

    def reset(self):
        """Reset (clean) caches. Should be called every time after epochs (as to not hog memory)."""
        self._cache = {}
        self._cache_args = {}

    def get_train_class_means_nums(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get the class means in all embeddings for the train dataset.

        :return: Dict of layer_id to torch tensor of size (num_classes, embedding_shape).
        """
        func_id = self.get_train_class_means_nums
        if func_id in self._cache.keys():
            assert (wrapped_model, dataset) == self._cache_args[func_id], "Got different arguments when calling cache, is this intended use of the cache? Did you reset?"
        else:
            class_means, class_nums = self._calc_class_means_nums(wrapped_model, dataset.train_loader, dataset.num_classes, dataset.is_one_hot)

            self._cache[func_id] = class_means, class_nums
            self._cache_args[func_id] = (wrapped_model, dataset)

        return self._cache[func_id]

    def get_test_class_means_nums(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get the class means in all embeddings for the test dataset.

        :return: Dict of layer_id to torch tensor of size (num_classes, embedding_shape).
       """
        func_id = self.get_test_class_means_nums
        if func_id in self._cache.keys():
            assert (wrapped_model, dataset) == self._cache_args[func_id], "Got different arguments when calling cache, is this intended use of the cache? Did you reset?"
        else:
            class_means, class_nums = self._calc_class_means_nums(wrapped_model, dataset.test_loader, dataset.num_classes, dataset.is_one_hot)

            self._cache[func_id] = class_means, class_nums
            self._cache_args[func_id] = (wrapped_model, dataset)

        return self._cache[func_id]

    @staticmethod
    def calc_global_mean(class_means: Dict[str, torch.Tensor], class_num_samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get the global mean of the activations in each layer from their mean and num_samples.

        :return: Dict of layer_id to torch tensor of size (1, *embedding_shape)
       """
        global_mean = {}
        for layer_name, layer_class_mean in class_means.items():
            global_mean[layer_name] = (layer_class_mean.transpose(0, -1) @ class_num_samples.view(-1, 1)
                                       / torch.sum(class_num_samples)).transpose(0, -1)
        return global_mean

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
    print(cache.get_test_class_means_nums(exp.wrapped_model, exp.dataset)['layer3'].size())
    print(cache.get_test_class_means_nums(exp.wrapped_model, exp.dataset)['layer3'].size())
    cache.reset()
    print(cache.get_test_class_means_nums(exp.wrapped_model, exp.dataset)['layer3'].size())


if __name__ == '__main__':
    _test_cache()
