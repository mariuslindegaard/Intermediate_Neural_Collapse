import torch
import Models
from DatasetWrapper import DatasetWrapper

from collections import defaultdict
from typing import Dict


class Measurer:
    def __init__(self):
        pass

    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper) -> Dict[str, float]:
        raise NotImplementedError("Measures should overwrite this method!")


class TraceMeasure(Measurer):
    def measure(self, wrapped_model: Models.ForwardHookedOutput, dataset: DatasetWrapper) -> Dict[str, float]:
        data_loader = dataset.train_loader
        measurements = defaultdict(int)

        device = next(wrapped_model.parameters()).device
        for inputs, targets in data_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            preds, embeddings = wrapped_model(inputs)  # embeddings: Dict[Hashable, torch.Tensor]

            for layer_name, activations in embeddings.items():
                measurements[layer_name] += torch.sum(torch.linalg.norm(activations))  # TODO(marius): Verify calculation (norm over sample, sum over batch)

        for layer_name in measurements.keys():
            measurements[layer_name] /= len(data_loader.dataset)

        return measurements
