import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
import tqdm
import yaml
from collections import OrderedDict
import pandas as pd

import Logger
from DatasetWrapper import DatasetWrapper
from OptimizerWrapper import OptimizerWrapper
import Models
import Measurer


class Experiment:
    _config_params: Dict
    dataset: DatasetWrapper
    wrapped_model: Models.ForwardHookedOutput
    wrapped_optimizer: OptimizerWrapper
    logger: Logger.Logger
    measures: Dict[str, callable]

    def __init__(self, config_path):
        with open(config_path, "r") as config_file:
            self._config_params = yaml.safe_load(config_file)
        model_cfg        = self._config_params['Model']  # noqa:E221
        data_cfg         = self._config_params['Data']  # noqa:E221
        logging_cfg      = self._config_params['Logging']  # noqa:E221
        optimizer_cfg    = self._config_params['Optimizer']  # noqa:E221
        measurements_cfg = self._config_params['Measurements']

        # Create logger
        self.logger = Logger.Logger(logging_cfg, config_path)
        self.logger.copy_config_to_dir()

        # Instantiate model and dataset
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.wrapped_model = Models.get_model(model_cfg)  # TODO(Marius): Support other models
        self.wrapped_model.base_model.to(device)
        self.dataset = DatasetWrapper(data_cfg)

        # Instantiate optimizer
        self.wrapped_optimizer = OptimizerWrapper(self.wrapped_model, optimizer_cfg)  # TODO(marius): Make sure LR scheduler supports checkpointing

        # Get all relevant measures
        self.measures = {measurement_str: getattr(Measurer, measurement_str)
                         for measurement_str in measurements_cfg['measures']}

    def _train_single_epoch(self):
        """Train the model on the specified dataset"""
        device = next(iter(self.wrapped_model.parameters())).device
        self.wrapped_model.train()

        optimizer = self.wrapped_optimizer.optimizer
        loss_function = self.wrapped_optimizer.criterion

        # Iterate over batches to train a single epoch
        pbar_batch = tqdm.tqdm(self.dataset.train_loader, position=1)
        for batch_index, (inputs, targets) in enumerate(pbar_batch):
            # Load data
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets if self.dataset.is_one_hot else F.one_hot(targets, num_classes=self.dataset.num_classes).float()
            targets_class_idx = torch.argmax(targets, dim=-1)

            # Calculate loss and do gradient step
            optimizer.zero_grad()
            preds, embeddings = self.wrapped_model(inputs)
            loss: torch.Tensor = loss_function(preds, targets)
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()

            pbar_batch.set_description(f'Loss: {loss.item():0.4f}\tLR: {optimizer.param_groups[0]["lr"]:0.6f}')

    def train(self):
        # TODO(marius): Implement checkpointing start
        # todo(marius): Implement tensorboard writer (if needed)
        epoch = 0
        pbar_epoch = tqdm.tqdm(range(epoch, self.wrapped_optimizer.max_epochs), position=0, leave=True)
        for epoch in pbar_epoch:
            # todo(marius): Implement warmup training (if wanting to copy cifar_100 repo exactly)

            self._train_single_epoch()

            self.wrapped_optimizer.lr_scheduler.step()

            # TODO(marius): Implement saving checkpoints

    def do_measurements_on_checkpoints(self, pbar_pos_offset=0):
        """Do measurements over all checkpoints saved"""
        model_path_list = self.logger.get_all_saved_model_paths()
        for model_checkpoint_path in tqdm.tqdm(model_path_list, position=pbar_pos_offset):
            self.wrapped_model, epoch, _ = self.logger.load_model(model_checkpoint_path, ret_model=self.wrapped_model)
            self.do_measurements()

    def do_measurements(self):
        """Do the intended measurements on the model"""
        measurement_dict = self.measures
        shared_cache = Measurer.SharedMeasurementVars()

        all_measurements = OrderedDict()
        for measurement_id, measurer in measurement_dict.items():
            measurement_result_df: pd.DataFrame = measurer.measure(
                measurer, self.wrapped_model, self.dataset, shared_cache=shared_cache
            )
            assert 'value' in measurement_result_df.columns, "Measurement dataframe must contain 'value' field."
            all_measurements[measurement_id] = measurement_result_df

        self.logger.write_to_measurements(all_measurements)

        # pbar_batch = tqdm.tqdm(self.dataset.train_loader, position=1+pbar_pos_offset, leave=False, ncols=None)
        # pbar_batch.set_description(f'')
        # pbar_batch.close()


def _test_measurer():
    config_path = "../config/default.yaml"
    exp = Experiment(config_path)
    exp.do_measurements()


def _test_training():
    config_path = "../config/default.yaml"
    exp = Experiment(config_path)
    exp.train()


if __name__ == "__main__":
    _test_training()
    # _test_measurer()
