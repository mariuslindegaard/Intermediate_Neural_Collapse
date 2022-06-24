import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional
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
        self.logger = Logger.Logger(logging_cfg, config_path, use_existing=True)
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
        pbar_batch = tqdm.tqdm(self.dataset.train_loader, position=1, leave=False)
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

            correct = torch.argmax(preds, dim=-1).eq(targets_class_idx).sum().item()

            pbar_batch.set_description(f'Loss: {loss.item():0.4G}  LR: {optimizer.param_groups[0]["lr"]:0.2G} Acc: {correct/len(inputs):0.3G}')

    def train(self):
        # TODO(marius): Assertion check that configs are equal
        if self.logger.get_all_saved_model_paths():
            latest_checkpoint_path = self.logger.get_all_saved_model_paths()[-1]
            _, start_epoch, _ = self.logger.load_model(
                latest_checkpoint_path, ret_model=self.wrapped_model, ret_optimizer=self.wrapped_optimizer
            )
            if start_epoch >= self.wrapped_optimizer.max_epochs:
                print("Model already trained, skipping training.")
                return
            print("Starting from ")
        else:
            start_epoch = 0

        # todo(marius): Implement tensorboard writer (if needed)
        pbar_epoch = tqdm.tqdm(range(start_epoch, self.wrapped_optimizer.max_epochs),
                               initial=start_epoch, total=self.wrapped_optimizer.max_epochs, position=0)
        for epoch in pbar_epoch:
            # todo(marius): Implement warmup training (if wanting to copy cifar_100 repo exactly)
            if epoch in self.logger.log_epochs:
                self.logger.save_model(self.wrapped_model, epoch, wrapped_optimizer=self.wrapped_optimizer)

            self._train_single_epoch()

            self.wrapped_optimizer.lr_scheduler.step()

        self.logger.save_model(self.wrapped_model, self.wrapped_optimizer.max_epochs, wrapped_optimizer=self.wrapped_optimizer)

    def do_measurements_on_checkpoints(self, pbar_pos_offset=0):
        """Do measurements over all checkpoints saved"""
        model_path_list = self.logger.get_all_saved_model_paths()
        for model_checkpoint_path in tqdm.tqdm(model_path_list, position=pbar_pos_offset):
            self.wrapped_model, epoch, _ = self.logger.load_model(model_checkpoint_path, ret_model=self.wrapped_model)
            self.do_measurements(epoch=epoch)

    def do_measurements(self, epoch: Optional[int] =None):
        """Do the intended measurements on the model

        :param epoch: Which epoch to assign to the measurements for this model (w/ parameters)"""
        measurement_dict = self.measures
        shared_cache = Measurer.SharedMeasurementVars()

        all_measurements = OrderedDict()
        for measurement_id, measurer in measurement_dict.items():
            measurement_result_df: pd.DataFrame = measurer.measure(
                measurer, self.wrapped_model, self.dataset, shared_cache=shared_cache
            )
            assert 'value' in measurement_result_df.columns, "Measurement dataframe must contain 'value' field."
            if epoch is not None:
                measurement_result_df.insert(0, 'epoch', epoch)
            all_measurements[measurement_id] = measurement_result_df

        self.logger.write_to_measurements(all_measurements)

        # pbar_batch = tqdm.tqdm(self.dataset.train_loader, position=1+pbar_pos_offset, leave=False, ncols=None)
        # pbar_batch.set_description(f'')
        # pbar_batch.close()


def _test_measurer():
    config_path = "../config/debug.yaml"
    exp = Experiment(config_path)
    exp.do_measurements()


def _test_training():
    config_path = "../config/default.yaml"
    exp = Experiment(config_path)
    exp.train()
    print("Running measurements!")
    exp.do_measurements_on_checkpoints()


if __name__ == "__main__":
    _test_training()
    # _test_measurer()
