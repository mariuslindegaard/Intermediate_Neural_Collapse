import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import yaml

from typing import Dict, Any, Iterator, List

def filter_configs(base_dir: str, required_params: Dict[str, Dict[str, Any]]) -> Iterator[str]:
    """Get all run directories in base_dir with configs matching required_params

    :param base_dir: Directory in which the runs are placed (contains timestamped dirs)
    :param required_params: Filter based on requiring these parameters to match the run config
    :return: An iterator over paths to directories matching this required config.
    """
    for run_dirname in os.listdir(base_dir):
        if run_dirname == 'latest' or run_dirname[0] == '.':
            continue
        cfg_path = os.path.join(base_dir, run_dirname, 'config.yaml')
        with open(cfg_path, 'r') as config_file:
            config_params = yaml.safe_load(config_file)

        # Assuming all config files are two layers deep (max)
        # Check if required_params is a subset of config_params
        for key, val in config_params.items():
            if key not in required_params.keys():
                continue

            if type(val) is dict:
                if required_params[key].items() <= val.items():
                    continue
                else:
                    break
            else:
                if required_params[key] == val:
                    continue
                else:
                    break
        else:  # Runs if no break
            run_dir = os.path.join(base_dir, run_dirname)
            yield run_dir


def plot_runs(base_dir):

    run_config_params = dict(
        Model={'model-name': 'resnet18'},
        Data={'dataset-id': 'cifar10'},
        Logging={},
        Measurements={},
    )
    relevant_measures = ['TraceMeasure']

    x_param = 'layer_name'
    hue_param = 'trace'

    for run_dir in filter_configs(base_dir, run_config_params):
        for measure in relevant_measures:
            # TODO(marius): Check if measurements file exists and warn+continue if not.
            measure_df = pd.read_csv(os.path.join(run_dir, 'measurements', measure + '.csv'))
            sns.lineplot(data=measure_df, x=x_param, y='value',
                         hue=hue_param)
            plt.show()
            print(measure_df)


def _test():
    sns.set_theme(style='darkgrid')
    root_dir = '/home/marius/mit/research/NN_layerwise_analysis'
    plot_runs(root_dir+'/logs/default')


if __name__ == '__main__':
    _test()
