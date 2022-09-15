import torch
import torch.nn.functional as F
import torch.utils.data
import Models
from DatasetWrapper import DatasetWrapper
import pandas as pd
import numpy as np
import scipy.sparse.linalg
import tqdm

from collections import defaultdict, OrderedDict
from typing import Dict, Tuple, List, Any, Union
import warnings

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
            shared_cache = SharedMeasurementVarsCache()
        wrapped_model.eval()  # TODO(marius): Might throw error. If not: Change in TraceMeasure too
        device = next(wrapped_model.parameters()).device

        dataset_splits = {
            'train': dataset.train_loader,
            'test': dataset.test_loader,
        }

        out: List[Dict[str, Any]] = []  # Output to return. List of different value entries, one for each datapoint.
        for split_id, data_loader in tqdm.tqdm(dataset_splits.items(), leave=False, desc='  Splits: '):
            num_samples, correct = 0, 0
            for inputs, targets in tqdm.tqdm(data_loader, leave=False, desc=f'    {split_id} batches: '):
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
            shared_cache = SharedMeasurementVarsCache()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device
        data_loader = dataset.train_loader

        class_trace_sums: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))  # Dict of, for each layer a dict of class number to total norm.

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)

        for inputs, targets in tqdm.tqdm(data_loader, leave=False, desc='  '):
            inputs, targets = inputs.to(device), targets.to(device)
            preds, embeddings = wrapped_model(inputs)  # embeddings: Dict[Hashable, torch.Tensor]
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets

            for class_idx, class_batch_indexes in enumerate(utils.class_idx_iterator(one_hot_targets)):
                for layer_name, activations in embeddings.items():
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
            shared_cache = SharedMeasurementVarsCache()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)

        total_activation_var: Dict[str, torch.Tensor] = OrderedDict()  # \Sigma_w
        # NCC_match_net = 0

        # M = torch.stack(mean).T  # Mean of classes before layer

        for inputs, targets in tqdm.tqdm(dataset.train_loader, leave=False, desc='  '):
            inputs, targets = inputs.to(device), targets.to(device)

            embeddings: Dict[str, torch.Tensor]
            preds, embeddings = wrapped_model(inputs)
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
            # class_idx_targets = torch.argmax(targets, dim=-1) if dataset.is_one_hot else targets

            for layer_name, activations in embeddings.items():
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
            cdnv = var_sums / torch.clamp(2*mean_diffs**2, min=1E-15)  # Clamp to avoid div by 0
            cdnv.fill_diagonal_(0)
            sum_cdnv = torch.sum(cdnv).item()

            out.append({'value': sum_cdnv, 'layer_name': layer_name})

        return pd.DataFrame(out)


class NC1Measure(Measurer):
    """Measure NC metrics"""

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVarsCache()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)

        # Get the class-wise covariances and sum them together to get total within-class covariance
        class_cov_within = shared_cache.get_train_class_covariance(wrapped_model, dataset)
        cov_within = {
            layer_name: (layer_class_cov_within * class_num_samples.unsqueeze(-1).unsqueeze(-1).to('cpu')) / torch.sum(class_num_samples).to('cpu')
            for layer_name, layer_class_cov_within in class_cov_within.items()
        }

        # Calculate NC1-condition and add to output
        out: List[Dict[str, Any]] = []
        for layer_name in tqdm.tqdm(cov_within.keys(), desc='  NC1, calc. metric', leave=False):
            layer_cov_within = cov_within[layer_name].to('cpu')
            S_within = torch.mean(layer_cov_within, axis=0).cpu().numpy()

            rel_class_means = (class_means[layer_name] - global_mean[layer_name]).flatten(start_dim=1).to('cpu')
            layer_cov_between = torch.matmul(rel_class_means.T, rel_class_means) / dataset.num_classes  # TODO(marius): Verify calculation
            S_between = layer_cov_between.cpu().numpy()
            eigvecs, eigvals, _ = scipy.linalg.svd(S_between)
            inv_eigvals = eigvals ** -1
            inv_eigvals[dataset.num_classes-1:] = 0  # Only use the first C-1 eigvals, since the relative between-class covariance is rank C-1
            inv_S_between = eigvecs @ np.diag(inv_eigvals) @ eigvecs.T  # Will get divide by 0 for first epochs, it is fine

            nc1_value = np.sum(np.trace(S_within @ inv_S_between))  # \Sigma_w @ \Sigma_b^-1
            out.append({'value': nc1_value, 'layer_name': layer_name})

        return pd.DataFrame(out)


class SingularValues(Measurer):
    """Singular values and their cumulative sum in weight matrix. Relative to total."""

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        # if shared_cache is None:
        #     shared_cache = SharedMeasurementVars()

        # wrapped_model.base_model.eval()
        # device = next(wrapped_model.parameters()).device

        # Calculate
        out: List[Dict[str, Any]] = []
        for layer_name in tqdm.tqdm(wrapped_model.output_layers, desc='  VarExpl.', leave=False):
            if not layer_name.endswith('fc'):
                continue

            # Get the model weights
            fc_layer = utils.rgetattr(wrapped_model.base_model, layer_name)
            try:
                weights = fc_layer.weight
            except AttributeError as e:
                warnings.warn(f"Module: {layer_name}, {fc_layer}\ndoes not have a 'weight' parameter. Make sure it is a fc-layer.")
                continue
            weights = weights.detach().to('cpu').numpy()

            U_w, S_w, Vh_w = scipy.linalg.svd(weights)  # weights == U_w @ "np.diag(S_w).reshape(weights.shape) (padded)" Vh_w

            rel_sigmas_sum = np.cumsum(S_w) / np.sum(S_w)
            rel_sigmas = S_w / np.sum(S_w)
            for idx, (rel_sigma, rel_sigma_sum) in enumerate(zip(rel_sigmas, rel_sigmas_sum)):
                out.append({'value': rel_sigma, 'sigma_idx': idx, 'layer_name': layer_name, 'sum': False})
                out.append({'value': rel_sigma_sum, 'sigma_idx': idx, 'layer_name': layer_name, 'sum': True})

        return pd.DataFrame(out)


class ActivationCovSVs(Measurer):
    """Singular values of activation covariance matrix."""

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVarsCache()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)
        classwise_cov_within = shared_cache.get_train_class_covariance(wrapped_model, dataset)

        # Calculate for each layer and each class
        out: List[Dict[str, Any]] = []
        for layer_name in tqdm.tqdm(wrapped_model.output_layers, desc='  ActivationCovSVs.', leave=False):
            total_within_sigmas = None
            for class_idx, class_cov_within in enumerate(classwise_cov_within[layer_name]):
                # U_w, S_w, Vh_w = torch.linalg.svd(class_cov_within.detach())  # weights == U_w @ "np.diag(S_w).reshape(weights.shape) (padded)" Vh_w
                S_w = torch.linalg.svdvals(class_cov_within.detach())

                rel_sigmas_sum = np.cumsum(S_w) / np.sum(S_w)
                rel_sigmas = S_w / np.sum(S_w)
                for idx, (rel_sigma, rel_sigma_sum) in enumerate(zip(rel_sigmas, rel_sigmas_sum)):
                    out.append({'value': rel_sigma, 'sigma_idx': idx, 'class_idx': class_idx, 'layer_name': layer_name, 'type': 'within_single', 'sum': False})
                    out.append({'value': rel_sigma_sum, 'sigma_idx': idx, 'class_idx': class_idx, 'layer_name': layer_name, 'type': 'within_single', 'sum': True})

                if total_within_sigmas is None:
                    total_within_sigmas = torch.zeros_like(rel_sigmas_sum).cpu()
                total_within_sigmas += rel_sigmas_sum.to_cpu() * class_num_samples[class_idx].cpu()

            rel_sigmas_sum = np.cumsum(total_within_sigmas) / np.sum(total_within_sigmas)
            rel_sigmas = total_within_sigmas / np.sum(total_within_sigmas)
            for idx, (rel_sigma, rel_sigma_sum) in enumerate(zip(rel_sigmas, rel_sigmas_sum)):
                out.append({'value': rel_sigma, 'sigma_idx': idx, 'class_idx': -1, 'layer_name': layer_name, 'type': 'within_sum', 'sum': False})
                out.append({'value': rel_sigma_sum, 'sigma_idx': idx, 'class_idx': -1, 'layer_name': layer_name, 'type': 'within_sum', 'sum': True})

            layer_rel_class_means = (class_means[layer_name] - global_mean[layer_name]).flatten(start_dim=1).cpu()
            layer_cov_between = torch.matmul(layer_rel_class_means.T, layer_rel_class_means).cpu() / dataset.num_classes

            S_w = torch.linalg.svdvals(layer_cov_between)
            rel_sigmas_sum = np.cumsum(S_w) / np.sum(S_w)
            rel_sigmas = total_within_sigmas / np.sum(S_w)
            for idx, (rel_sigma, rel_sigma_sum) in enumerate(zip(rel_sigmas, rel_sigmas_sum)):
                out.append({'value': rel_sigma, 'sigma_idx': idx, 'class_idx': -1, 'layer_name': layer_name, 'type': 'between', 'sum': False})
                out.append({'value': rel_sigma_sum, 'sigma_idx': idx, 'class_idx': -1, 'layer_name': layer_name, 'type': 'between', 'sum': True})

        # TODO(marius): Bugtest this!

        return pd.DataFrame(out)


class MLPSVDMeasure(Measurer):
    """Measure MLP SVD metrics"""

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVarsCache()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)
        classwise_cov_within = shared_cache.get_train_class_covariance(wrapped_model, dataset)

        # Calculate NC1-condition and add to output
        out: List[Dict[str, Any]] = []
        for layer_name in tqdm.tqdm(classwise_cov_within.keys(), desc='  SVD, calc. metric', leave=False):
            if not layer_name.endswith('fc'):
                continue

            # Get class means and covariance
            layer_rel_class_means = (class_means[layer_name] - global_mean[layer_name]).flatten(start_dim=1).to('cpu')
            layer_classwise_cov_within = classwise_cov_within[layer_name].to('cpu')
            layer_cov_between = torch.matmul(layer_rel_class_means.T, layer_rel_class_means) / dataset.num_classes

            # Get the model weights
            fc_layer = utils.rgetattr(wrapped_model.base_model, layer_name)
            try:
                weights = fc_layer.weight
            except AttributeError as e:
                warnings.warn(f"Module: {layer_name}, {fc_layer}\ndoes not have a 'weight' parameter. Make sure it is a fc-layer.")
                continue
            weights = weights.detach().to('cpu').numpy()

            U_w, S_w, Vh_w = scipy.linalg.svd(weights)  # weights == U_w @ "np.diag(S_w).reshape(weights.shape) (padded)" Vh_w

            num_from_weights = 100
            num_from_class = 10
            for class_idx, class_cov_within in enumerate(layer_classwise_cov_within):
                U_c, S_c, Vh_c = scipy.linalg.svd(class_cov_within)
                Vh_w_sliced = Vh_w[:num_from_weights]
                Vh_c_sliced = Vh_c[:num_from_class]  # == U_c[:, :num_from_class]

                # Logging inner products of class mean and weight svd
                class_mean = class_means[layer_name][class_idx].detach().to('cpu').numpy()
                class_mean /= np.linalg.norm(class_mean)  # Normalize class mean

                """ For using rows instead of class means
                for w_idx, w_row in enumerate(weights):
                    w_row /= np.linalg.norm(w_row)
                    correlation = class_mean.T @ w_row
                    out.append({'value': correlation, 'layer_name': layer_name,
                                'l_type': -2, 'l_ord': w_idx,
                                'r_type': class_idx, 'r_ord': 'm',
                                })
                """

                for w_idx, w_singular_vec in enumerate(Vh_w_sliced):
                    if w_idx >= len(S_w):  # Don't evaluate if rank is lower than idx
                        continue
                    correlation = class_mean.T @ w_singular_vec
                    out.append({'value': correlation, 'layer_name': layer_name,
                                'l_type': -1, 'l_ord': w_idx,
                                'r_type': class_idx, 'r_ord': 'm',
                                })
                """ For visualizing singular vectors
                    # Plot singular vector:
                    import matplotlib.pyplot as plt
                    # import pdb; pdb.set_trace()
                    digit_vec = w_singular_vec
                    plt.imshow(digit_vec.reshape(32, 32), cmap='gray')
                    plt.title(f"Singular vector {w_idx} with $\\sigma={S_w[w_idx]:.5G}$\nOut-svec: {np.round(U_w[:,w_idx], 2)}")
                    plt.savefig(f'tmp_w_{w_idx}')
                continue
                """

                # Logging inner products of class covariance and weight svd
                corr = Vh_w_sliced @ Vh_c_sliced.T
                for (w_idx, c_idx), w_c_corr in np.ndenumerate(corr):
                    if w_idx >= len(S_w) or c_idx >= len(S_c):  # Don't evaluate if rank is lower than idx
                        continue
                    out.append({'value': w_c_corr, 'layer_name': layer_name,
                                'l_type': -1, 'l_ord': w_idx,
                                'r_type': class_idx, 'r_ord': c_idx,
                                })
        # exit()

        return pd.DataFrame(out)


class AngleBetweenSubspaces(Measurer):
    """Measure angle between subspaces of first 10 singular vectors of weights and class means"""

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVarsCache()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        # global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)
        # classwise_cov_within = shared_cache.get_train_class_covariance(wrapped_model, dataset)

        # Calculate NC1-condition and add to output
        out: List[Dict[str, Any]] = []
        for layer_name in tqdm.tqdm(class_means.keys(), desc='  AngleBet.Subspc., calculating', leave=False):
            if not layer_name.endswith('fc'):
                continue

            # layer_rel_class_means = (class_means[layer_name] - global_mean[layer_name]).flatten(start_dim=1).to('cpu')
            # layer_classwise_cov_within = classwise_cov_within[layer_name].to('cpu')
            # layer_cov_between = torch.matmul(layer_rel_class_means.T, layer_rel_class_means) / dataset.num_classes
            layer_class_means = class_means[layer_name]

            # Get the model weights
            fc_layer = utils.rgetattr(wrapped_model.base_model, layer_name)
            try:
                weights = fc_layer.weight
            except AttributeError as e:
                warnings.warn(f"Module: {layer_name}, {fc_layer}\ndoes not have a 'weight' parameter. Make sure it is a fc-layer.")
                continue
            weights = weights.detach().to('cpu').numpy()

            U_w, S_w, Vh_w = scipy.linalg.svd(weights)  # weights == U_w @ "np.diag(S_w).reshape(weights.shape) (padded)" Vh_w
            U_m, S_m, Vh_m = scipy.linalg.svd(layer_class_means.T)  # l_c_m.T is (d x C)

            # U, S, Vh = scipy.linalg.svd(Vh_w @ U_m)
            S = scipy.linalg.svdvals(Vh_w @ U_m)

            rel_sigmas_sum = np.cumsum(S) / np.sum(S)
            rel_sigmas = S / np.sum(S)
            for idx, (rel_sigma, rel_sigma_sum) in enumerate(zip(rel_sigmas, rel_sigmas_sum)):
                out.append({'value': rel_sigma, 'sigma_idx': idx, 'layer_name': layer_name, 'sum': False})
                out.append({'value': rel_sigma_sum, 'sigma_idx': idx, 'layer_name': layer_name, 'sum': True})

        return pd.DataFrame(out)


class SharedMeasurementVarsCache:
    """Shared cache of often-used computations used for measurements.

    E.g. class means. This allows them to be calculated only once per epoch!
    """
    COV_MAX_BATCH_SIZE = 64  # <- Maximum batch size to use when doing memory-heavy calculations.
    COV_IGNORE_LAYER_IDS = ['', 'model', 'model.flatten', 'conv1', 'bn1', 'relu', 'maxpool']  # TODO(marius): Make less hardcoded
    COV_NUM_LAYERS = 0  # Number of layers to use. 0 for all, -x for the last x, +x for the first x.

    def __init__(self):
        self._cache: Dict[callable, Union[Tuple[Dict[str, torch.Tensor], torch.Tensor], Dict[str, torch.Tensor]]] = {}
        self._cache_args: Dict[callable, Tuple[Models.ForwardHookedOutput, DatasetWrapper]] = {}

        self.layer_slice_size = defaultdict(lambda: self.COV_MAX_BATCH_SIZE)

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

    def get_train_class_covariance(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper) -> Dict[str, torch.Tensor]:
        """Get the activation covariances within each class

        :return: Dict of layer_id to torch tensor of size (num_classes, embedding_width, embedding_width).
       """
        func_id = self.get_train_class_covariance
        if func_id in self._cache.keys():
            assert (wrapped_model, dataset) == self._cache_args[func_id], "Got different arguments when calling cache, is this intended use of the cache? Did you reset?"
        else:
            class_covariances = self._calc_train_class_covariance(wrapped_model, dataset)

            self._cache[func_id] = class_covariances
            self._cache_args[func_id] = (wrapped_model, dataset)

        return self._cache[func_id]

    @staticmethod
    def _calc_class_means_nums(wrapped_model: Models.ForwardHookedOutput, data_loader: torch.utils.data.DataLoader, num_classes: int, is_one_hot: bool) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        device = next(iter(wrapped_model.parameters())).device
        class_trace_sums = {layer_name: None for layer_name in wrapped_model.output_layers}
        class_nums = torch.zeros((num_classes,)).to(device)

        for inputs, targets in tqdm.tqdm(data_loader, leave=False, desc=f"  Cache; class means"):
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

    def _calc_train_class_covariance(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper) -> Dict[str, torch.Tensor]:
        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = self.get_train_class_means_nums(wrapped_model, dataset)

        cov_within: Dict[str, torch.Tensor] = OrderedDict()  # \Sigma_w
        # NCC_match_net = 0

        # M = torch.stack(mean).T  # Mean of classes before layer
        # Go through layers. check only the last one if
        layers = wrapped_model.output_layers
        if self.COV_NUM_LAYERS != 0:
            layers = layers[:self.COV_NUM_LAYERS] if self.COV_NUM_LAYERS > 0 else layers[self.COV_NUM_LAYERS:]
        layer_pbar = tqdm.tqdm(layers, leave=False)
        for layer_name in layer_pbar:
            layer_pbar.set_description(f"  Cache; covariances: {layer_name:16}")
            # Ignore the largest layers of resnet18
            if layer_name in self.COV_IGNORE_LAYER_IDS:  # TODO(marius): Make only catch relevant net, not all
                continue

            batch_pbar = tqdm.tqdm(dataset.train_loader, leave=False)
            for full_inputs, full_targets in batch_pbar:
                use_cpu_for_batch = False
                batch_pbar.set_description(f"     Batches; Slices: {self.layer_slice_size[layer_name]}")
                # Iterate only over slices of the batch, not the full batch
                for inputs, targets in utils.slice_to_smaller_batch(full_inputs, full_targets, batch_size=self.layer_slice_size[layer_name]):
                    inputs, targets = inputs.to(device), targets.to(device)

                    embeddings: Dict[str, torch.Tensor]
                    preds, embeddings = wrapped_model(inputs)
                    one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
                    # class_idx_targets = torch.argmax(targets, dim=-1) if dataset.is_one_hot else targets

                    flat_activations = torch.flatten(embeddings[layer_name], start_dim=1)
                    if layer_name not in cov_within.keys():
                        activation_size = flat_activations.size()[-1]
                        cov_within[layer_name] = torch.zeros((dataset.num_classes, activation_size, activation_size), device='cpu')

                    for class_idx, class_batch_indexes in enumerate(utils.class_idx_iterator(one_hot_targets)):
                        if not len(class_batch_indexes):  # Continue if no images classified to this class
                            continue
                        class_activations = flat_activations[class_batch_indexes, :].detach()

                        # update within-class cov
                        rel_class_activations = (class_activations - class_means[layer_name][class_idx].reshape(1, -1))
                        if use_cpu_for_batch:
                            rel_class_activations = rel_class_activations.detach().to('cpu')
                        # Catch out-of-memory errors, reduce batch size for this layer and continue with CPU for this slice.
                        try:
                            cov = torch.matmul(rel_class_activations.unsqueeze(-1),  # B CHW 1
                                               rel_class_activations.unsqueeze(1))  # B 1 CHW
                        except RuntimeError as e:
                            if "CUDA out of memory" not in str(e):
                                raise e
                            # warnings.warn("CUDA out of memory")
                            # Use CPU for the rest of this batch, and reduce the "safe" batch size
                            use_cpu_for_batch = True
                            old_bs = self.layer_slice_size[layer_name]
                            self.layer_slice_size[layer_name] = (old_bs+1) // 2  # Making sure the bs never becomes 0
                            batch_pbar.set_description(f"     Batches; Slices: {old_bs}(cpu) -> {(old_bs+1) // 2}")

                            rel_class_activations = rel_class_activations.detach().to('cpu')
                            cov = torch.matmul(rel_class_activations.unsqueeze(-1),  # B CHW 1
                                               rel_class_activations.unsqueeze(1))  # B 1 CHW

                        cov_within[layer_name][class_idx] += torch.sum(cov, dim=0).detach().to('cpu')

        # Make cov_within an average instead of a sum
        for layer_name, cov_within_sum in cov_within.items():
            cov_within[layer_name] = cov_within_sum / class_num_samples.unsqueeze(-1).unsqueeze(-1).to('cpu')  # TODO(marius): Use bessels correction?

        return cov_within


def _test_cache():
    from Experiment import Experiment
    config_path = "../config/default.yaml"
    exp = Experiment(config_path)
    # exp.do_measurements()
    cache = SharedMeasurementVarsCache()
    # print(cache.get_test_class_means_nums(exp.wrapped_model, exp.dataset)['layer3'].size())
    # print(cache.get_test_class_means_nums(exp.wrapped_model, exp.dataset)['layer3'].size())
    cache.reset()
    # print(cache.get_test_class_means_nums(exp.wrapped_model, exp.dataset)['layer3'].size())


if __name__ == '__main__':
    _test_cache()
