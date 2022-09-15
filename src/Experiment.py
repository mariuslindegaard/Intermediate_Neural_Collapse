import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional
import tqdm
import yaml
from collections import OrderedDict
import pandas as pd
import warnings

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

        # Instantiate model and dataset
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = DatasetWrapper(data_cfg)
        self.wrapped_model = Models.get_model(model_cfg, self.dataset)
        self.wrapped_model.base_model.to(device)

        # Instantiate optimizer
        self.wrapped_optimizer = OptimizerWrapper(self.wrapped_model, optimizer_cfg)

        # Get all relevant measures
        self.measures = {measurement_str: getattr(Measurer, measurement_str)()
                         for measurement_str in measurements_cfg['measures']}

        # Copy config to correct file. Do last so any initalization errors get thrown first.
        self.logger.copy_config_to_dir()

    def _train_single_epoch(self):
        """Train the model on the specified dataset"""
        device = next(iter(self.wrapped_model.parameters())).device
        self.wrapped_model.train()

        optimizer = self.wrapped_optimizer.optimizer
        loss_function = self.wrapped_optimizer.criterion

        # Iterate over batches to train a single epoch
        pbar_batch = tqdm.tqdm(self.dataset.train_loader, leave=False)
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

            pbar_batch.set_description(f'Loss: {loss.item():5.3E}  LR: {optimizer.param_groups[0]["lr"]:.2G} Acc: {correct/len(inputs): <6.3G}')

    def train(self):
        # Start from checkpoint or skip training if previously trained
        if self.logger.get_all_saved_model_paths():
            latest_checkpoint_path = sorted(self.logger.get_all_saved_model_paths())[-1]
            _, start_epoch, _ = self.logger.load_model(
                latest_checkpoint_path, ret_model=self.wrapped_model, ret_optimizer=self.wrapped_optimizer
            )
            if start_epoch >= self.wrapped_optimizer.max_epochs:
                print("Model already trained, skipping training.")
                return
            print("Starting from ")
        else:
            start_epoch = 0

        # todo(marius): Implement tensorboard writer (if needed), or just writing to logs.
        pbar_epoch = tqdm.tqdm(range(start_epoch, self.wrapped_optimizer.max_epochs),
                               initial=start_epoch, total=self.wrapped_optimizer.max_epochs,
                               desc='Epochs',
                               )
        for epoch in pbar_epoch:
            # todo(marius): Implement warmup training (if wanting to copy cifar_100 repo exactly)
            if epoch in self.logger.log_epochs:
                self.logger.save_model(self.wrapped_model, epoch, wrapped_optimizer=self.wrapped_optimizer)

            self._train_single_epoch()

            self.wrapped_optimizer.lr_scheduler.step()

        self.logger.save_model(self.wrapped_model, self.wrapped_optimizer.max_epochs, wrapped_optimizer=self.wrapped_optimizer)

    def do_measurements_on_checkpoints(self):
        """Do measurements over all checkpoints saved"""
        model_path_list = list(reversed(self.logger.get_all_saved_model_paths()))
        for model_checkpoint_path in tqdm.tqdm(model_path_list, desc='Checkpoints'):
            self.wrapped_model, epoch, _ = self.logger.load_model(model_checkpoint_path, ret_model=self.wrapped_model)
            self.do_measurements(epoch=epoch)

    def do_measurements(self, epoch: Optional[int] = None):
        """Do the intended measurements on the model

        :param epoch: Which epoch to assign to the measurements for this model (w/ parameters)
        """
        measurement_dict = self.measures
        shared_cache = Measurer.SharedMeasurementVarsCache()

        all_measurements = OrderedDict()
        pbar_measurements = tqdm.tqdm(measurement_dict.items(), leave=None)
        for measurement_id, measurer in pbar_measurements:
            pbar_measurements.set_description(f"Measure: {measurement_id}")
            measurement_result_df: pd.DataFrame = measurer.measure(
                self.wrapped_model, self.dataset, shared_cache=shared_cache
            )

            # Check that measurements are good:
            assert 'value' in measurement_result_df.columns or len(measurement_result_df) == 0,\
                f"Measurement dataframe for {measurement_id} must contain 'value' field but is {measurement_result_df}"
            if not len(measurement_result_df):
                raise Exception(f"Dataframe for {measurement_id} does not contain data!")
                # warnings.warn(f"Dataframe for {measurement_id} does not contain data!")
                # continue  # Do not add measurement

            if epoch is not None:
                measurement_result_df.insert(0, 'epoch', epoch)

            all_measurements[measurement_id] = measurement_result_df

        self.logger.write_to_measurements(all_measurements)


def _test():
    import os
    import sys
    # config_path = "../config/default.yaml"
    # config_path = "../config/vgg16.yaml"
    # run_on_cbcl = True

    config_path = "../config/debug.yaml"

    print(sys.executable)
    if '/cbcl/cbcl01/lindegrd/miniconda3/envs/' in sys.executable:
        cbcl_base = '/cbcl/cbcl01/lindegrd/NN_layerwise/src/'
        config_path = os.path.join(cbcl_base, config_path)

    print(f"Using config at {config_path}")
    exp = Experiment(config_path)
    exp.train()
    print("Running measurements!")
    exp.do_measurements_on_checkpoints()


if __name__ == "__main__":
    _test()
