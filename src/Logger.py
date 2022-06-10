import warnings
from dataclasses import dataclass
import os
import subprocess
import shutil
from typing import List, Dict, Union, DefaultDict, Hashable
from collections import OrderedDict, defaultdict
import datetime




class SaveDirs:
    def __init__(self, dirname: str):
        root_dir = ""
        try:
            root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('ascii')[:-1]
        except OSError as e:  # TODO(marius): Make exception catching less general
            warnings.warn("Finding git root directory failed with the following error message:\n"+str(e))
            root_dir = ""
            warnings.warn(f"Using '{root_dir}' as root directory")

        self._base = os.path.join(
            root_dir,
            dirname,
            datetime.datetime.now().isoformat(timespec='minutes')
        )
        while True:
            try:
                os.makedirs(self._base)
            except OSError as e:
                if e.errno != 17:
                    warnings.warn("Got other OSError than expected for file exists:")
                    warnings.warn(str(e))
                self._base += '_'
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


class Logger:
    save_dirs: SaveDirs  # Save directories
    config_path: str  # Path to config file

    def __init__(self, logging_cfg: Dict, config_path: str):
        self.save_dirs = SaveDirs(logging_cfg['save-dir'])
        self.config_path = config_path
        self.logging_path = None

    def write_to_log(self, logs: Dict[Dict[Hashable, float]]):  # TODO(marius): Add test to verify all data is in correct order
        """Log all logging values to file.

        Assumes values in logs are the same eaach time, with the same ordering. This b.c. it writes a single line to a
        csv file.
        """
        raise NotImplementedError("Needs updating for new measurements")  # TODO(marius)
        if self.logging_path is None:
            self.logging_path = os.path.join(self.save_dirs.base, "log.csv")
            with open(self.logging_path, "w") as log_file:
                log_file.write(",".join(map(str, logs.keys())))
                log_file.write("\n")

        with open(self.logging_path, "a") as log_file:
            log_file.write(",".join(map(str, logs.values())))
            log_file.write("\n")

    def copy_config_to_dir(self):  # TODO(marius): Make copy to tmp at start, and copy to permanent when finished
        shutil.copy(self.config_path, os.path.join(self.save_dirs.base, "config.yaml"), follow_symlinks=True)
        # Set latest run to this run
        self.force_symlink(
            os.path.split(self.save_dirs.base)[1],  # Use relative path
            os.path.join(self.save_dirs.base, "../latest"),
        )

    @staticmethod
    def force_symlink(target, link_name):
        try:
            temp_link = link_name + ".tmp"
            # os.remove(temp_link)
            os.symlink(target, temp_link)
            os.rename(temp_link, link_name)
        except OSError as e:
            warnings.warn("Failed to create symlink!")
            raise e
    # Handle critical errors


if __name__ == "__main__":
    pass