import warnings
import os
import subprocess
import shutil
import filecmp
from typing import Dict, Union, Hashable, Optional, Tuple, List, Iterable
import datetime
import pandas as pd

import Models
import OptimizerWrapper

import torch


class SaveDirs:
    def __init__(self, dirname: str, timestamp_subdir: bool = True, use_existing: bool = False):
        """Initialize, find root directory and create base directory.

        :param dirname: Save-directory path (from project (i.e. git) root directory)
        :param timestamp_subdir: Use subfolder in the save-directory path with timestamp of the run.
        :param use_existing:
        """
        # Find root directory
        try:
            root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('ascii')[:-1]
        except (OSError, subprocess.CalledProcessError) as e:  # TODO(marius): Make exception catching less general
            warnings.warn("Finding git root directory failed with the following error message:\n"+str(e))
            root_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), os.pardir
            ))
            warnings.warn(f"Using '{root_dir}' as root directory")

        # Set base directory
        self._base = os.path.join(
            root_dir,
            dirname,
        )

        # Define subdirectory and append to base
        if timestamp_subdir:
            if timestamp_subdir is True:
                timestamp_subdir = datetime.datetime.now().isoformat(timespec='minutes')
            self._base = os.path.join(self._base, timestamp_subdir)


        # Make sure directory does not already exist, and if it does handle it. (Behavior based on use_existing)
        idx = 0
        while True:
            try:
                os.makedirs(self._base)
            except OSError as e:
                if e.errno != 17:
                    warnings.warn("Got other OSError than expected for file exists:")
                    warnings.warn(str(e))

                if use_existing:
                    break

                if self._base.endswith(f'_{idx}'):
                    self._base = self._base[:-len(str(idx))]
                else:
                    self._base += '_'
                idx += 1
                self._base = self._base + str(idx)
            else:
                break

    @property
    def base(self) -> str:
        return self._base

    @property
    def measurements(self) -> str:
        path = os.path.join(self.base, 'measurements')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def data(self) -> str:
        path = os.path.join(self.base, 'data')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def models(self):
        path = os.path.join(self.base, 'models')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def plots(self):
        path = os.path.join(self.base, 'plots')
        if not os.path.exists(path):
            os.makedirs(path)
        return path


class Logger:
    save_dirs: SaveDirs  # Save directories
    config_path: str  # Path to config file
    log_epochs: Iterable[int]

    def __init__(self, logging_cfg: Dict, config_path: str, timestamp_subdir=False, use_existing=False):
        self.save_dirs = SaveDirs(logging_cfg['save-dir'], timestamp_subdir=timestamp_subdir, use_existing=use_existing)
        self.config_path = config_path
        self.logging_path = None
        if 'log-epochs' in logging_cfg.keys():
            self.log_epochs = logging_cfg['log-epochs']
        else:
            self.log_epochs = list(range(0, 10)) + list(range(10, 2000, logging_cfg['log-interval']))

    def write_to_log(self, logs: Dict[Hashable, float]):  # TODO(marius): Add test to verify all data is in correct order
        """Log all logging values to file.

        :param logs: Dict of values to log.

        Assumes keys in logs are the same with identical ordering each time, with the same ordering.
        This b.c. it writes a single line to a csv file with the key indexing only on the top line.
        """
        if self.logging_path is None:
            self.logging_path = os.path.join(self.save_dirs.base, "log.csv")
            with open(self.logging_path, "w") as log_file:
                log_file.write(",".join(map(str, logs.keys())))
                log_file.write("\n")

        with open(self.logging_path, "a") as log_file:
            log_file.write(",".join(map(str, logs.values())))
            log_file.write("\n")

    def write_to_measurements(self, measurements: Dict[Hashable, pd.DataFrame]):
        """Writes to the measurement files from a dict of dataframes of measurements.

        :param measurements: Specifies {'measure_type': df_1, ...} where df contains a column 'values' and
            all other columns are metadata for that specific value. measure_type gives filename.

        """
        for measure_str, measurement_df in measurements.items():
            # Split df into heading and contained data
            df_csv_str = measurement_df.to_csv(index=False)
            start_second_line_idx = df_csv_str.find('\n') + 1
            heading, data = df_csv_str[:start_second_line_idx], df_csv_str[start_second_line_idx:]

            filepath = os.path.join(self.save_dirs.measurements, f'{measure_str}.csv')
            # Write heading if file does not exist
            if not os.path.exists(filepath):
                with open(filepath, "w") as f:
                    f.write(heading)

            # Append new data to file, verifying that heading is the same as for the file in general.
            with open(filepath, 'a+') as f:
                f.seek(0)
                file_heading = f.readline()
                assert heading == file_heading, f"Columns of df is not the same as in file, {heading}{file_heading}."
                f.seek(0, 2)  # Seek to end of file
                f.write(data)

    def save_model(self, wrapped_model: Models.ForwardHookedOutput, epoch: int,
                   wrapped_optimizer: OptimizerWrapper.OptimizerWrapper = None) -> str:
        """Save pytorch model with epoch number"""
        save_path = os.path.join(self.save_dirs.models, f'{epoch:0>3}.tar')
        torch.save(dict(
            wrapped_model_state_dict=wrapped_model.state_dict(),
            epoch=epoch,
            wrapped_optimizer_state_dict=wrapped_optimizer.state_dict() if wrapped_optimizer is not None else None
        ), save_path)
        return save_path

    def load_model(self, path_specification: Union[str, int], ret_model: Models.ForwardHookedOutput,
                   ret_optimizer: Optional[OptimizerWrapper.OptimizerWrapper] = None) -> \
            Tuple[Models.ForwardHookedOutput, int, Optional[OptimizerWrapper.OptimizerWrapper]]:
        """Load model from path.

        NOTE: ret_model and ret_optimizer must have same architecture as the saved models.
        Specifically: Use the wrappers.

        :param path_specification: Path to model file.
        :param ret_model: Model to be loaded into. Must have same underlying model architechture as input to save_model.
        :param ret_optimizer: Optimizer to apply the (possibly saved) state dict to.
        :return: Tuple of (Model_state_dict, epoch_number, optimizer_state_dict (if saved, otherwise None))
        """
        if type(path_specification) is int:
            save_path = os.path.join(self.save_dirs.models, f'{path_specification}.tar')
        else:
            save_path = path_specification
        checkpoint = torch.load(save_path)

        ret_model.load_state_dict(checkpoint['wrapped_model_state_dict'])
        epoch = checkpoint['epoch']
        if ret_optimizer is not None:
            assert checkpoint['wrapped_optimizer_state_dict'] is not None, f"Found no saved optimizer state dict to load in {save_path}"
            ret_optimizer.load_state_dict(checkpoint['wrapped_optimizer_state_dict'])

        return ret_model, epoch, ret_optimizer

    def get_all_saved_model_paths(self) -> List[str]:
        """Return a tuple of paths to all saved models for this run."""
        model_paths = []
        for saved_path in sorted(os.listdir(self.save_dirs.models)):
            if saved_path == 'latest':
                continue
            model_paths.append(os.path.join(
                self.save_dirs.models, saved_path
            ))
        return model_paths


    @staticmethod
    def _torch_tensor_to_float(value: Union[float, torch.Tensor]):
        """Take a torch tensor to a float, and a float to a float."""
        if type(value) is torch.Tensor:
            return value.item()
        return value

    def copy_config_to_dir(self, throw_if_existing_differs=True):
        """Copy the config file to the working directory and set the current dir to 'latest'

        :param throw_if_existing_differs: Whether to throw an error if the existing config file in that location differs
        """
        target_path = os.path.join(self.save_dirs.base, "config.yaml")
        # If config exists and differs, throw or warn
        if os.path.exists(target_path):
            config_is_equal = filecmp.cmp(self.config_path, target_path)
            if not config_is_equal:
                if throw_if_existing_differs:
                    raise FileExistsError(f"Config file at {target_path} exists and differs from input config at {self.config_path}!")
                else:
                    warnings.warn(f"Config file at {target_path} exists and differs from input config at {self.config_path}!")

        shutil.copy(self.config_path, target_path, follow_symlinks=True)
        # Set latest run to this run
        self.force_symlink(
            os.path.split(self.save_dirs.base)[1],  # Use relative path
            os.path.join(self.save_dirs.base, "../latest"),
        )

    @staticmethod
    def force_symlink(target, link_name):
        """Create a symlink, throwing an error if not possible"""
        temp_link = link_name + ".tmp"
        try:
            os.symlink(target, temp_link)
            os.rename(temp_link, link_name)
        except OSError as e:
            warnings.warn("Failed to create symlink!")
            if e.errno == 21:
                warnings.warn(f"The 'latest'-directory is supposed to be a symlink, not a directory"
                              f"\n\tDelete the 'latest' directory at {link_name} to fix error.")
            if os.path.exists(temp_link):
                os.remove(temp_link)
            raise e


if __name__ == "__main__":
    pass
