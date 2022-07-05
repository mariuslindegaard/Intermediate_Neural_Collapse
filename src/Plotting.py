import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import yaml

from Logger import SaveDirs

from typing import Dict, Any, Iterator, List


def filter_configs(base_dir: str, required_params: Dict[str, Dict[str, Any]]) -> Iterator[str]:
    """Get all run directories in base_dir with configs matching required_params

    :param base_dir: Directory in which the runs are placed (contains timestamped dirs)
    :param required_params: Filter based on requiring these parameters to match the run config
    :return: An iterator over paths to directories matching this required config.
    """
    dirs = sorted(map(lambda d: d.path, filter(lambda d: d.is_dir(), os.scandir(base_dir))))

    for run_dirname in dirs:
        if run_dirname.endswith('latest') or os.path.split(run_dirname)[-1][0] == '.':
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
    """The base plotting function to copy and modify."""

    run_config_params = dict(
        # Model={'model-name': 'resnet18'},
        # Data={'dataset-id': 'cifar10'},
        # Optimizer={},
        # Logging={},
        # Measurements={},
    )
    relevant_measures = ['AccuracyMeasure']

    x_param = 'epoch'
    hue_param = 'split'
    style_param = None

    for measure in relevant_measures:
        print(f"Plotting {measure}:")
        for run_dir in filter_configs(base_dir, run_config_params):
            print(f"\t{run_dir}", end=', ')
            savedir = SaveDirs(run_dir, timestamp_subdir=False, use_existing=True)
            # TODO(marius): Merge dataframes and plot
            fig = plt.figure(figsize=(8, 8))
            # TODO(marius): Check if measurements file exists and warn+continue if not.
            measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))

            # selection = measure_df['epoch'].isin([0, 10, 20, 40, 70, 100, 160, 200])
            selection = measure_df['epoch'] != -1

            sns.lineplot(data=measure_df[selection], x=x_param, y='value',
                         hue=hue_param, style=style_param)
            # plt.yscale('log')
            plt.title(f"{measure} over {x_param} for \n{os.path.split(savedir.base)[-1]}")
            plt.tight_layout()
            savepath = os.path.join(savedir.plots, measure + '.pdf')
            print(f"saving to {savepath}")
            plt.savefig(savepath)
            plt.show()



def plot_runs_rel_trace(base_dir):
    """The base plotting function to copy and modify."""

    run_config_params = dict(
        # Model={'model-name': 'resnet18'},
        # Data={'dataset-id': 'cifar10'},
        # Optimizer={},
        # Logging={},
        # Measurements={},
    )
    relevant_measures = {
        'TraceMeasure': dict(x='layer_name', hue='epoch', style='trace', style_order=['sum', 'between', 'within']),
        'CDNVMeasure': dict(x='layer_name', hue='epoch')}

    for measure, plot_config in relevant_measures.items():
        print(f"Plotting {measure}:")
        for run_dir in filter_configs(base_dir, run_config_params):
            print(f"\t{run_dir}", end=', ')
            savedir = SaveDirs(run_dir, timestamp_subdir=False, use_existing=True)
            if measure == 'TraceMeasure':
                fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex='all')
                plt.sca(axes[0])
            else:
                fig = plt.figure(figsize=(10, 8))

            measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))

            # selection = measure_df['epoch'].isin([0, 10, 20, 40, 70, 100, 160, 200])
            # selection = measure_df['epoch'].isin([0, 40, 160, 200])
            selection = (measure_df['epoch'] != -1) & (measure_df['layer_name'] != 'model')

            # Plot absolute traces
            sns.lineplot(data=measure_df[selection], y='value', **plot_config)
            plt.title(f"{measure} over {plot_config['x']} for \n{os.path.split(savedir.base)[-1]}")
            plt.yscale('log')

            # Plot relative trace
            if measure == "TraceMeasure":
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
                sns.lineplot(data=df[selection][df[selection]['trace'] != 'sum'], y='value', **plot_config)
                # sns.lineplot(data=df[selection][df[selection]['trace'] == 'between'], y='value', **plot_config)
                plt.yscale('log')

            plt.xticks(rotation=90)

            plt.tight_layout()
            savepath = os.path.join(savedir.plots, measure + '.pdf')
            print(f"saving to {savepath}")
            plt.savefig(savepath)
            plt.show()


def _test():
    sns.set_theme(style='darkgrid')
    root_dir = '/home/marius/mit/research/NN_layerwise_analysis'
    # plot_runs(root_dir+'/logs/base_run')
    plot_runs_rel_trace(root_dir+'/logs/base_run')


if __name__ == '__main__':
    _test()
