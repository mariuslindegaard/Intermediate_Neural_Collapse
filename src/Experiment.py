import torch
import torch.nn as nn

from typing import Dict
import tqdm
import yaml
from collections import OrderedDict

import Logger
from DatasetWrapper import DatasetWrapper
import Models
import Measurer


class Experiment:
    _config_params: Dict
    dataset: DatasetWrapper
    wrapped_model: Models.ForwardHookedOutput

    def __init__(self, config_path):
        with open(config_path, "r") as config_file:
            self._config_params = yaml.safe_load(config_file)
        model_cfg        = self._config_params['Model']  # noqa:E221
        data_cfg         = self._config_params['Data']  # noqa:E221
        logging_cfg      = self._config_params['Logging']  # noqa:E221
        measurements_cfg = self._config_params['Measurements']

        self.logger = Logger.Logger(logging_cfg, config_path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.wrapped_model = Models.get_resnet18_model(model_cfg)  # TODO(Marius): Support other models
        self.wrapped_model.base_model.to(device)

        self.dataset = DatasetWrapper(data_cfg)

        # TODO(marius): Get a hold on what is happening with this config copying
        self.logger.copy_config_to_dir()

        self.measures = {measurement_str: getattr(Measurer, measurement_str) for measurement_str in measurements_cfg['measures']}

    def do_measurements(self):
        """Do the intended measurements on the model"""
        measurement_dict = self.measures
        shared_cache = Measurer.SharedMeasurementVars()

        all_measurements = OrderedDict()
        for measurement_id, measurer in measurement_dict.items():
            measurement_result = measurer.measure(measurer, self.wrapped_model, self.dataset, share_cache=shared_cache)
            all_measurements[measurement_id] = measurement_result

        self.logger.write_to_measurements(all_measurements)

        # pbar_batch = tqdm.tqdm(self.dataset.train_loader, position=1+pbar_pos_offset, leave=False, ncols=None)
        # pbar_batch.set_description(f'')
        # pbar_batch.close()


def _test():
    config_path = "../config/default.yaml"
    exp = Experiment(config_path)
    exp.do_measurements()



if __name__ == "__main__":
    _test()
