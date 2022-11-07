import torch
import torch.nn.functional as F
import torch.utils.data
import Models
from DatasetWrapper import DatasetWrapper
import pandas as pd
import numpy as np
import scipy.sparse.linalg
import tqdm
import tensorly.decomposition

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


class Accuracy(Measurer):
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


class Traces(Measurer):
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


class CDNV(Measurer):
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

        for inputs, targets in tqdm.tqdm(dataset.train_loader, leave=False, desc='  CDNV, Calculating'):
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
                    total_activation_var[layer_name][class_idx] += torch.sum(rel_class_activations ** 2).detach().cpu()

        out: List[Dict[str, Any]] = []
        for layer_name, total_var in total_activation_var.items():

            var_per_sample = total_var.cpu() / class_num_samples.cpu()
            rel_class_means = torch.flatten(class_means[layer_name] - global_mean[layer_name], start_dim=1).cpu()

            mean_diffs = torch.cdist(rel_class_means, rel_class_means)
            var_sums = var_per_sample.reshape(-1, 1) + var_per_sample
            cdnv = var_sums / torch.clamp(2*mean_diffs**2, min=1E-15)  # Clamp to avoid div by 0
            cdnv.fill_diagonal_(0)
            sum_cdnv = torch.sum(cdnv).item() / (dataset.num_classes**2 - dataset.num_classes)

            out.append({'value': sum_cdnv, 'layer_name': layer_name})

        return pd.DataFrame(out)


class NC1(Measurer):
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


class WeightSVs(Measurer):
    """Singular values and their cumulative sum in weight matrix. Relative to total."""

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        # if shared_cache is None:
        #     shared_cache = SharedMeasurementVars()

        # wrapped_model.base_model.eval()
        # device = next(wrapped_model.parameters()).device

        # Calculate
        out: List[Dict[str, Any]] = []
        for layer_name in tqdm.tqdm(wrapped_model.output_layers, desc='  VarExpl.', leave=False):
            layer_obj = utils.rgetattr(wrapped_model.base_model, layer_name)
            if isinstance(layer_obj, torch.nn.Linear):  # For MLP models and classifier in VGG/ResNet
                # Get layer weights
                try:
                    weights = layer_obj.weight
                except AttributeError as e:
                    warnings.warn(f"Module: {layer_name}, {layer_obj}\ndoes not have a 'weight' parameter. Make sure it is a fc-layer.")
                    continue
                weights = weights.detach().cpu().numpy()


            elif isinstance(layer_obj, torch.nn.Conv2d):  # For all convlayers in VGG (and others)
                # Get layer weights
                try:
                    weights = layer_obj.weight
                except AttributeError as e:
                    warnings.warn(f"Module: {layer_name}, {layer_obj}\ndoes not have a 'weight' parameter. Make sure it is a fc-layer.")
                    continue

                # Preprocess by reshaping etc.
                weights = weights.flatten(start_dim=1).cpu().detach().numpy()  # .transpose(1, 0)

            else:
                continue


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
                S_w = torch.linalg.svdvals(class_cov_within.detach()).to('cpu').numpy()

                rel_sigmas_sum = np.cumsum(S_w) / np.sum(S_w)
                rel_sigmas = S_w / np.sum(S_w)
                for idx, (rel_sigma, rel_sigma_sum) in enumerate(zip(rel_sigmas, rel_sigmas_sum)):
                    out.append({'value': rel_sigma, 'sigma_idx': idx, 'class_idx': class_idx, 'layer_name': layer_name, 'type': 'within_single', 'sum': False})
                    out.append({'value': rel_sigma_sum, 'sigma_idx': idx, 'class_idx': class_idx, 'layer_name': layer_name, 'type': 'within_single', 'sum': True})

                if total_within_sigmas is None:
                    total_within_sigmas = np.zeros_like(rel_sigmas_sum)
                total_within_sigmas += rel_sigmas_sum * class_num_samples[class_idx].cpu().numpy()

            rel_sigmas_sum = np.cumsum(total_within_sigmas) / np.sum(total_within_sigmas)
            rel_sigmas = total_within_sigmas / np.sum(total_within_sigmas)
            for idx, (rel_sigma, rel_sigma_sum) in enumerate(zip(rel_sigmas, rel_sigmas_sum)):
                out.append({'value': rel_sigma, 'sigma_idx': idx, 'class_idx': -1, 'layer_name': layer_name, 'type': 'within_sum', 'sum': False})
                out.append({'value': rel_sigma_sum, 'sigma_idx': idx, 'class_idx': -1, 'layer_name': layer_name, 'type': 'within_sum', 'sum': True})

            layer_rel_class_means = (class_means[layer_name] - global_mean[layer_name]).flatten(start_dim=1).cpu()
            layer_cov_between = torch.matmul(layer_rel_class_means.T, layer_rel_class_means).cpu() / dataset.num_classes

            S_w = torch.linalg.svdvals(layer_cov_between).to('cpu').numpy()
            rel_sigmas_sum = np.cumsum(S_w) / np.sum(S_w)
            rel_sigmas = total_within_sigmas / np.sum(S_w)
            for idx, (rel_sigma, rel_sigma_sum) in enumerate(zip(rel_sigmas, rel_sigmas_sum)):
                out.append({'value': rel_sigma, 'sigma_idx': idx, 'class_idx': -1, 'layer_name': layer_name, 'type': 'between', 'sum': False})
                out.append({'value': rel_sigma_sum, 'sigma_idx': idx, 'class_idx': -1, 'layer_name': layer_name, 'type': 'between', 'sum': True})

        return pd.DataFrame(out)


class ActivationStableRank(Measurer):
    """Calculate (a version of) stable rank of activations.

    Estimates, for the relative activation matrix A (activations minus class-mean for each class):
        ||A||_F^2 / ||A||_2^2 = sum_i sigma_i^2 / max_i sigma_i^2
    """

    # OMEGA = 0.5
    MAX_ITERS = 20
    COS_ANGLE_DIFF_THRESHOLD = 0.99

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVarsCache()
        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)
        # classwise_cov_within = shared_cache.get_train_class_covariance(wrapped_model, dataset)


        # Calculate \sum_i \sigma^2_i = Tr(A^T A) = \sum_i a_i^T a_i (Squared Frobenius-norm?) of covariance matrix
        # (note a_i = h_i - \mu_i for a class c):
        ## Indexed by layer_name then class_idx
        frobenius_sq: Dict[str, torch.Tensor] = {layer_name: torch.zeros(dataset.num_classes, device='cpu')
                                                 for layer_name in class_means.keys()}

        for inputs, targets in tqdm.tqdm(dataset.train_loader, leave=False, desc='  ActivationStableRank [0/2], Frobenius'):
            inputs, targets = inputs.to(device), targets.to(device)

            embeddings: Dict[str, torch.Tensor]
            preds, embeddings = wrapped_model(inputs)
            one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
            # class_idx_targets = torch.argmax(targets, dim=-1) if dataset.is_one_hot else targets

            for layer_name, activations in embeddings.items():
                for class_idx, class_batch_indexes in enumerate(utils.class_idx_iterator(one_hot_targets)):
                    if not len(class_batch_indexes):  # Continue if no images classified to this class
                        continue
                    class_activations = activations[class_batch_indexes, :].detach()
                    rel_class_activations = (class_activations - class_means[layer_name][class_idx]).flatten(start_dim=1)  # n x d

                    # Calculate inner product of all relative activations belonging to that class, add to frobeinus_sq
                    frobenius_sq[layer_name][class_idx] += torch.trace(rel_class_activations @ rel_class_activations.T).detach().cpu()  #


        prev_layer_eigvecs: Dict[str, torch.tensor] = {layer_name: torch.ones_like(class_mean).flatten(start_dim=1) / len(class_mean[0])**(1/2)
                                                 for layer_name, class_mean in class_means.items()}

        # for inputs, targets in tqdm.tqdm(dataset.train_loader, leave=False, desc='  ActivationStableRank [0/2], Frobenius'):
        # TODO(marius): Implement progress bar
        pbar = tqdm.tqdm(range(ActivationStableRank.MAX_ITERS), leave=False, desc='  ActivationStableRank [1/2], Eigvecs')
        for epoch_iters in pbar:

            curr_layer_eigvecs = {layer_name: torch.zeros_like(prev) for layer_name, prev in prev_layer_eigvecs.items()}

            for inputs, targets in tqdm.tqdm(dataset.train_loader, leave=False, desc='    Batches'):
                inputs, targets = inputs.to(device), targets.to(device)

                embeddings: Dict[str, torch.Tensor]
                preds, embeddings = wrapped_model(inputs)
                one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets
                # class_idx_targets = torch.argmax(targets, dim=-1) if dataset.is_one_hot else targets

                for layer_name, activations in embeddings.items():
                    for class_idx, class_batch_indexes in enumerate(utils.class_idx_iterator(one_hot_targets)):
                        if not len(class_batch_indexes):  # Continue if no images classified to this class
                            continue
                        # import pdb; pdb.set_trace()
                        class_activations = activations[class_batch_indexes, :].detach()
                        rel_class_activations = (class_activations - class_means[layer_name][class_idx]).flatten(start_dim=1)

                        # Calculate inner products of relative activations with previous iteration, multiply with vector and add to new eigvec
                        inner_prods = rel_class_activations @ prev_layer_eigvecs[layer_name][class_idx]  # (n,)
                        curr_layer_eigvecs[layer_name][class_idx] += (inner_prods @ rel_class_activations).detach() / class_num_samples[class_idx]

            # Lower bound on eigvals:
            curr_layer_eigvals = {layer_name: eigvecs.norm(dim=1)
                            for layer_name, eigvecs in curr_layer_eigvecs.items()}
            curr_layer_eigvecs_normed = {layer_name: eigvecs / curr_layer_eigvals[layer_name].unsqueeze(1)
                                         for layer_name, eigvecs in curr_layer_eigvecs.items()}


            # Find the minimum cosine angle difference between current and previous eigvec estimate
            min_cos_angle = 1
            for layer_name, eigvecs_normed in curr_layer_eigvecs_normed.items():
                layer_min_cos_angle = torch.min(torch.diag(prev_layer_eigvecs[layer_name] @ eigvecs_normed.T))
                min_cos_angle = min(min_cos_angle, layer_min_cos_angle)

            # If the cosine-angle difference with the previous estimated eigvec is lower than the threshold, accept the estimate.
            if min_cos_angle > ActivationStableRank.COS_ANGLE_DIFF_THRESHOLD:
                break
            else:
                pbar.set_description(f'ActivationStableRank [1/2], Eigvecs, Angle diff: {min_cos_angle:.3G}')

            # Update old eigvecs to match new
            prev_layer_eigvecs = curr_layer_eigvecs_normed
            # for layer_name, prev_eigvecs in prev_layer_eigvecs.items():
            #     curr_eigvecs = curr_layer_eigvecs_normed[layer_name]
            #     omega = ActivationStableRank.OMEGA  # For faster convergence
            #     new_eigvec = curr_eigvecs*(1+omega) - omega*prev_eigvecs
            #     prev_layer_eigvecs[layer_name] = new_eigvec / new_eigvec.norm(dim=1, keepdim=True)

        # Calculate for each layer and each class
        out: List[Dict[str, Any]] = []
        for layer_name in tqdm.tqdm(wrapped_model.output_layers, desc='  ActivationStableRank, postproc.', leave=False):
            sq_stable_rank = frobenius_sq[layer_name].cpu() / (curr_layer_eigvals[layer_name]**2).cpu()  # Note: Sq. of frobenius over largest sq. SV)
            for class_idx, class_sq_stable_rank in enumerate(sq_stable_rank):
                out.append({'value': class_sq_stable_rank.item(), 'layer_name': layer_name, 'class_idx': class_idx})

        return pd.DataFrame(out)


class MLPSVD(Measurer):
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
            # layer_cov_between = torch.matmul(layer_rel_class_means.T, layer_rel_class_means) / dataset.num_classes

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
    """Measure angle between subspaces of convolutional networks"""

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVarsCache()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)

        out: List[Dict[str, Any]] = []
        for layer_name in tqdm.tqdm(class_means.keys(), desc='  ConvAngleBet.Subspc., calculating', leave=False):
            layer_obj = utils.rgetattr(wrapped_model.base_model, layer_name)
            if isinstance(layer_obj, torch.nn.Linear):  # For MLP models and classifier in VGG/ResNet
                # Get layer weights
                try:
                    weights = layer_obj.weight
                except AttributeError as e:
                    warnings.warn(f"Module: {layer_name}, {layer_obj}\ndoes not have a 'weight' parameter. Make sure it is a fc-layer.")
                    continue
                weights = weights.detach()

                # Decompose layer weights
                U_w, S_w, Vh_w = torch.linalg.svd(weights)  # weights == U_w @ "np.diag(S_w).reshape(weights.shape) (padded)" Vh_w

                # Decompose class means
                layer_class_means = class_means[layer_name]
                U_m, S_m, Vh_m = torch.linalg.svd(layer_class_means.T)  # l_c_m.T is (d x C)

                # Calculate principal angles
                rank = dataset.num_classes
                S = torch.linalg.svdvals(Vh_w[:rank] @ U_m[:, :rank]).to('cpu').numpy()
                avg_cos_angle = np.sum(S) / rank

                out.append({'value': avg_cos_angle.item(), 'rank': rank, 'layer_name': layer_name, 'layer_type': 'fc'})

            elif isinstance(layer_obj, torch.nn.Conv2d):  # For all convlayers in VGG (and others)
                # Get layer weights
                try:
                    weights = layer_obj.weight
                except AttributeError as e:
                    warnings.warn(f"Module: {layer_name}, {layer_obj}\ndoes not have a 'weight' parameter. Make sure it is a fc-layer.")
                    continue

                """
                # Preprocess by reshaping etc.
                weights = weights.flatten(-2, -1).transpose(1, 0).cpu().detach().numpy()
                layer_class_means = class_means[layer_name].flatten(-2, -1).permute(1, 2, 0).detach().cpu().numpy()
                
                # Decompose weights
                S_w, (U_w, V_w, X_w) = tensorly.decomposition.tucker(weights, rank=rank)
                
                # Decompose class means
                S_m, (U_m, V_m, X_m) = tensorly.decomposition.tucker(layer_class_means, rank=rank)
                
                S = scipy.linalg.svdvals(
                    U_w.T @ U_m  # U_w[:, :rank].T @ U_m[:, :rank]
                )
                """

                weights_tx = weights.transpose(1, 0).flatten(start_dim=1)  # ch_in x (C * h * w)
                features_tx = class_means[layer_name].transpose(1, 0).flatten(start_dim=1)  # ch_in x (ch_out * h * w)
                U_w,S_w,Vh_w = torch.linalg.svd(weights_tx.detach())
                U_m,S_m,Vh_m = torch.linalg.svd(features_tx.detach())  # Note: Sm is rank (C * h * w)

                # rank = dataset.num_classes
                max_rank = min(U_w.shape[1], U_m.shape[1])
                ranks = list(range(1, min(100, max_rank+1), 1)) + list(range(100, max_rank+1, dataset.num_classes))
                for rank in ranks:
                    S = torch.linalg.svdvals(U_w[:, :rank].t() @ U_m[:, :rank]).to('cpu').detach().numpy()
                    avg_cos_angle = np.sum(S) / rank
                    out.append({'value': avg_cos_angle.item(), 'rank': rank, 'layer_name': layer_name, 'layer_type': 'conv'})

            else:
                continue

            # layer_rel_class_means = (class_means[layer_name] - global_mean[layer_name]).flatten(start_dim=1).to('cpu')
            # layer_classwise_cov_within = classwise_cov_within[layer_name].to('cpu')
            # layer_cov_between = torch.matmul(layer_rel_class_means.T, layer_rel_class_means) / dataset.num_classes

            # U, S, Vh = scipy.linalg.svd(Vh_w @ U_m)

            # S_sum = np.cumsum(S) / 10  # TODO(marius): Remove "/10" and update NCPlotter._plot_angleBetweenSubspaces (i.e. remove "*10")

            # for idx, (sigma, sigma_sum) in enumerate(zip(S, S_sum)):
            #     out.append({'value': sigma, 'sigma_idx': idx, 'layer_name': layer_name, 'sum': False})
            #     out.append({'value': sigma_sum, 'sigma_idx': idx, 'layer_name': layer_name, 'sum': True})

        return pd.DataFrame(out)


'''
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
            weights = weights.detach()

            U_w, S_w, Vh_w = torch.linalg.svd(weights)  # weights == U_w @ "np.diag(S_w).reshape(weights.shape) (padded)" Vh_w
            U_m, S_m, Vh_m = torch.linalg.svd(layer_class_means.T)  # l_c_m.T is (d x C)

            # U, S, Vh = scipy.linalg.svd(Vh_w @ U_m)
            S = torch.linalg.svdvals(Vh_w[:dataset.num_classes] @ U_m[:, :dataset.num_classes]).to('cpu').numpy()

            S_sum = np.cumsum(S) / dataset.num_classes

            for idx, (sigma, sigma_sum) in enumerate(zip(S, S_sum)):
                out.append({'value': sigma, 'sigma_idx': idx, 'layer_name': layer_name, 'sum': False})
                out.append({'value': sigma_sum, 'sigma_idx': idx, 'layer_name': layer_name, 'sum': True})

        return pd.DataFrame(out)
'''

class ETF(Measurer):
    """Measure 'angle class means plus 1/(C-1)' and norms of relative class means.

    cos(angle)+1/(C-1) -> 0 in NC2, and expect norms to be the same length.
    """

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVarsCache()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        class_means, class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        global_mean = shared_cache.calc_global_mean(class_means, class_num_samples)
        # classwise_cov_within = shared_cache.get_train_class_covariance(wrapped_model, dataset)

        out: List[Dict[str, Any]] = []
        for layer_name in tqdm.tqdm(class_means.keys(), desc='  ETFAngle, calculating', leave=False):

            layer_rel_class_means = (class_means[layer_name] - global_mean[layer_name]).flatten(start_dim=1).to('cpu')
            # layer_classwise_cov_within = classwise_cov_within[layer_name].to('cpu')
            # layer_cov_between = torch.matmul(layer_rel_class_means.T, layer_rel_class_means) / dataset.num_classes
            # layer_class_means = class_means[layer_name]

            layer_rel_class_means_norms = torch.linalg.norm(layer_rel_class_means, dim=1)  # Class means in rows
            normed_layer_rel_class_means = torch.nn.functional.normalize(layer_rel_class_means, dim=1)  # Class means in rows

            class_means_cos_angles = normed_layer_rel_class_means @ normed_layer_rel_class_means.T
            class_means_cos_angles_plus_correction = class_means_cos_angles + 1 / (dataset.num_classes - 1)

            for l_class_idx in range(dataset.num_classes):
                out.append({'value': layer_rel_class_means_norms[l_class_idx].item(), 'layer_name': layer_name,
                            'l_ord': l_class_idx, 'r_ord': l_class_idx, 'type': 'norm'})
                for r_class_idx in range(dataset.num_classes):
                    if l_class_idx == r_class_idx:
                        continue
                    out.append({'value': class_means_cos_angles_plus_correction[l_class_idx][r_class_idx].item(),
                                'layer_name': layer_name,
                                'l_ord': l_class_idx, 'r_ord': r_class_idx, 'type': 'angle'})

        return pd.DataFrame(out)


class NCC(Measurer):
    """Measure accuracy of nearest-class mean classifier"""

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper, shared_cache=None) -> pd.DataFrame:
        if shared_cache is None:
            shared_cache = SharedMeasurementVarsCache()

        wrapped_model.base_model.eval()
        device = next(wrapped_model.parameters()).device

        train_class_means, train_class_num_samples = shared_cache.get_train_class_means_nums(wrapped_model, dataset)
        test_class_means, test_class_num_samples = shared_cache.get_test_class_means_nums(wrapped_model, dataset)

        dataset_splits = {
            'train': (dataset.train_loader, train_class_num_samples),
            'test': (dataset.test_loader, test_class_num_samples),  # Uses train class means for
        }
        out: List[Dict[str, Any]] = []
        for split_id, (data_loader, split_class_num_samples) in tqdm.tqdm(dataset_splits.items(), leave=False, desc='  NCC, splits: '):

            layer_class_wise_correct: Dict[str, torch.tensor] = defaultdict(lambda: torch.zeros(dataset.num_classes, device='cpu'))
            for inputs, targets in tqdm.tqdm(data_loader, desc=f'    {split_id} batches', leave=False):
                inputs, targets = inputs.to(device), targets.to(device)

                embeddings: Dict[str, torch.Tensor]
                preds, embeddings = wrapped_model(inputs)
                one_hot_targets = F.one_hot(targets, num_classes=dataset.num_classes) if not dataset.is_one_hot else targets

                # Calculate accuracy of nearest class mean classifier in each layer embedding
                for layer_name, activations in embeddings.items():
                    layer_class_means = train_class_means[layer_name].flatten(start_dim=1).to(device).unsqueeze(0).detach()
                    activations = activations.flatten(start_dim=1).detach()

                    dists = torch.cdist(activations, layer_class_means).squeeze()
                    one_hot_ncc_preds = F.one_hot(dists.argmin(dim=-1), num_classes=dataset.num_classes)
                    one_hot_correct = one_hot_targets * one_hot_ncc_preds

                    layer_class_wise_correct[layer_name] += torch.sum(one_hot_correct, dim=0).to('cpu')

            for layer_name, class_wise_correct in layer_class_wise_correct.items():
                # class_wise_accuracy = class_wise_correct / split_class_num_samples
                accuracy = class_wise_correct.sum() / split_class_num_samples.sum()
                out.append({
                    'value': accuracy.item(), 'layer_name': layer_name, 'split': split_id,
                })

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


SLOW_MEASURES = [
    'NC1',
    'ActivationCovSVs',
    'MLPSVD',
]

FAST_MEASURES = [
    'Accuracy',
    'Traces',  # Paper: NC1
    'CDNV',
    # 'NC1',
    'WeightSVs',
    # 'ActivationCovSVs',
    # 'MLPSVD',
    'AngleBetweenSubspaces',  # Paper: NC3
    # 'ConvAngleBetweenSubspaces',  # Paper: NC3
    'ETF',  # Paper: NC2
    'NCC',  # Paper: NC4
]

STABLERANK_MEASURE = [
    'ActivationStableRank',
]

ALL_MEASURES = FAST_MEASURES + STABLERANK_MEASURE + SLOW_MEASURES

if __name__ == '__main__':
    _test_cache()
