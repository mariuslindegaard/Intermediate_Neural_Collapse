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
        # Optimizer={},
        # Logging={},
        # Measurements={},
    )
    relevant_measures = ['TraceMeasure']

    x_param = 'layer_name'
    hue_param = 'epoch'
    style_param = 'trace'
    # hue_param, style_param = style_param, hue_param

    for measure in relevant_measures:
        for run_dir in filter_configs(base_dir, run_config_params):
            # TODO(marius): Merge dataframes and plot
            fig = plt.figure(figsize=(8, 8))
            # TODO(marius): Check if measurements file exists and warn+continue if not.
            measure_df = pd.read_csv(os.path.join(run_dir, 'measurements', measure + '.csv'))

            df = measure_df
            selection = df['epoch'].isin([0, 10, 20, 40, 70, 100, 160, 200])

            sns.lineplot(data=measure_df[selection], x=x_param, y='value',
                         hue=hue_param, style=style_param, style_order=['sum', 'between', 'within'])
            plt.yscale('log')
            plt.title(f"{measure} over {x_param} for \n{run_config_params}")
            plt.tight_layout()
            plt.savefig("traces.pdf")
            plt.show()


def plot_runs_rel_trace(base_dir):

    run_config_params = dict(
        Model={'model-name': 'resnet18'},
        Data={'dataset-id': 'cifar10'},
        # Optimizer={},
        Logging={'save-dir': 'logs/default'},
        # Measurements={},
    )
    relevant_measures = ['TraceMeasure']

    x_param = 'layer_name'
    hue_param = 'epoch'
    style_param = 'trace'
    # hue_param, style_param = style_param, hue_param

    for measure in relevant_measures:
        for run_dir in filter_configs(base_dir, run_config_params):
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex='all')
            plt.sca(axes[0])

            measure_df = pd.read_csv(os.path.join(run_dir, 'measurements', measure + '.csv'))

            # selection = measure_df['epoch'].isin([0, 10, 20, 40, 70, 100, 160, 200])
            # selection = measure_df['epoch'].isin([0, 40, 160, 200])
            selection = measure_df['epoch'] != -1

            # Plot absolute traces
            sns.lineplot(data=measure_df[selection], x=x_param, y='value',
                         hue=hue_param, style=style_param, style_order=['sum', 'between', 'within'])
            plt.yscale('log')
            plt.title(f"{measure} over {x_param} for \n{run_config_params}")

            # Plot relative trace
            plt.sca(axes[1])
            df = measure_df
            total_trace_sel = df['trace'] == 'sum'
            between_trace_sel = df['trace'] == 'between'
            within_trace_sel = df['trace'] == 'within'
            total_trace = df['value'].to_numpy()[total_trace_sel]
            df.at[between_trace_sel, 'value'] = df['value'].to_numpy()[between_trace_sel] / total_trace
            df.at[within_trace_sel, 'value'] = df['value'].to_numpy()[within_trace_sel] / total_trace
            # between_arr = df['value'].to_numpy()[between_trace_sel] / total_trace
            # within_arr = df['value'].to_numpy()[within_trace_sel] / total_trace

            sns.lineplot(data=df[selection][df[selection]['trace'] != 'sum'], x=x_param, y='value',
                         hue=hue_param, style=style_param, style_order=['sum', 'between', 'within'])

            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, 'plots', "traces.pdf"))
            plt.show()


def _test():
    sns.set_theme(style='darkgrid')
    root_dir = '/home/marius/mit/research/NN_layerwise_analysis'
    # plot_runs(root_dir+'/logs')
    plot_runs_rel_trace(root_dir+'/logs')


if __name__ == '__main__':
    _test()
