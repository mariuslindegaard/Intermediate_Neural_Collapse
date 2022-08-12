import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

import os
import yaml

from Logger import SaveDirs
import utils

from typing import Dict, Any, Iterator, List


def filter_configs(base_dir: str, required_params: Dict[str, Dict[str, Any]], recurse: bool = False) -> Iterator[str]:
    """Get all run directories in base_dir with configs matching required_params

    :param base_dir: Directory in which the runs are placed (contains timestamped dirs)
    :param required_params: Filter based on requiring these parameters to match the run config
    :param recurse: Recurse downwards through filepaths. Default False
    :return: An iterator over paths to directories matching this required config.
    """
    dirs = sorted(map(lambda d: d.path, filter(lambda d: d.is_dir(), os.scandir(base_dir))))

    for run_dirname in dirs:
        if run_dirname.endswith('latest') or os.path.split(run_dirname)[-1][0] == '.':
            continue
        cfg_path = os.path.join(base_dir, run_dirname, 'config.yaml')
        try:
            with open(cfg_path, 'r') as config_file:
                config_params = yaml.safe_load(config_file)
        except FileNotFoundError as e:  # If there is not immediate config file, either recurse or
            if recurse:
                for child_dir in filter_configs(base_dir=os.path.join(base_dir, run_dirname), required_params=required_params, recurse=recurse):
                    yield child_dir
            continue

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


def plot_runs(base_dir, run_config_params):
    """The base plotting function to copy and modify."""

    relevant_measures = {
        'AccuracyMeasure': dict(x='epoch', hue='split', style=None)
    }

    for measure, plot_config in relevant_measures.items():
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

            sns.lineplot(data=measure_df[selection], y='value', **plot_config)
            # plt.yscale('log')
            plt.title(f"{measure} over {plot_config['x']} for \n{os.path.split(savedir.base)[-1]}")
            plt.tight_layout()
            savepath = os.path.join(savedir.plots, measure + '.pdf')
            print(f"saving to {savepath}")
            plt.savefig(savepath)
            plt.show()


def plot_runs_svds(base_dir, run_config_params):
    """For plotting the SVD correlation matrices."""

    relevant_measures = {
        'MLPSVDMeasure': None  # dict(x='epoch', hue='split', style=None)
    }

    for measure, plot_config in relevant_measures.items():
        print(f"Plotting {measure}:")
        for run_dir in filter_configs(base_dir, run_config_params):
            print(f"\t{run_dir}", end=', ')
            savedir = SaveDirs(run_dir, timestamp_subdir=False, use_existing=True)
            # TODO(marius): Merge dataframes and plot
            # TODO(marius): Check if measurements file exists and warn+continue if not.
            measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))
            # sub_selection = (measure_df['l_ord'] <= 40) & (measure_df['r_ord'].isin(['m'] + list(map(str, range(4)))))
            sub_selection = (measure_df['l_ord'] <= 40) & (measure_df['r_ord'].isin(['m']))  #  & (measure_df['l_type'].isin([-2]))

            layers = measure_df['layer_name'].unique()
            epochs = measure_df['epoch'].unique()

            savepath = os.path.join(savedir.plots, measure)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            savepath = os.path.join(savepath, r'e{epoch}/{layer}.pdf')
            print(f"saving to {savepath}")

            for epoch in tqdm.tqdm(epochs, leave=False):
                print(f"Epoch: {epoch}")
                for layer in layers:
                    fig = plt.figure(figsize=(8, 6))
                    selection = (measure_df['epoch'] == epoch) & (measure_df['layer_name'] == layer) & sub_selection


                    corr_df = utils.corr_from_df(measure_df[selection])

                    sns.heatmap(corr_df, # .applymap(abs),
                                # cmap='vlag', vmin=-0.5, vmax=0.5,
                                cmap='inferno', # vmin=0.0, vmax=1,
                                )
                    plt.title(f"SVD correlation for {os.path.split(savedir.base)[-1]}:\n"
                              f"Epoch: {epoch}, layer: {layer}")
                    plt.tight_layout()
                    formatted_savepath = savepath.format(layer=layer, epoch=epoch)
                    if not os.path.exists(os.path.dirname(formatted_savepath)):
                        os.makedirs(os.path.dirname(formatted_savepath))
                    plt.savefig(formatted_savepath)
                    # plt.show()
                    plt.close()
            print()


def plot_runs_rel_trace(base_dir, run_config_params):
    """The base plotting function to copy and modify."""

    relevant_measures = {
        'TraceMeasure': dict(x='layer_name', hue='epoch', style='trace', style_order=['sum', 'between', 'within']),
        'CDNVMeasure': dict(x='layer_name', hue='epoch'),
        'NC1Measure': dict(x='layer_name', hue='epoch'),
    }

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
            selection = (measure_df['epoch'] != -1) & (measure_df['layer_name'] != 'model')  # & (measure_df['layer_name'].isin(['conv1', 'bn1', *[f'layer{i//2}.{i%2}' for i in range(2, 10)], 'avgpool', 'fc']))

            # Plot absolute traces
            ## If there is a hue-parameter and the x-axis has only one entry, replace the x-axis with the hue-parameter
            if 'hue' in plot_config.keys() and len(measure_df[selection][plot_config['x']].unique()) == 1:
                plot_config['x'] = plot_config['hue']
                del plot_config['hue']
            sns.lineplot(data=measure_df[selection], y='value', **plot_config)
            plt.title(f"{measure} over {plot_config['x']} for \n{os.path.split(savedir.base)[-1]}")
            plt.yscale('log')

            # Plot relative trace
            if measure == "TraceMeasure":
                plt.legend(loc='center left')
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
                plt.legend(loc='center left')

            plt.xticks(rotation=90)

            plt.tight_layout()
            savepath = os.path.join(savedir.plots, measure + '.pdf')
            print(f"saving to {savepath}")
            plt.savefig(savepath)
            plt.show()


def main(logs_parent_dir: str):
    """Run some standard plotting on the measurements. Prone to failure!!!"""
    sns.set_theme(style='darkgrid')
    run_config_params = dict(  # All parameters must match what is given here.
        # Model={'model-name': 'resnet18'},
        # Data={'dataset-id': 'cifar10'},
        # Optimizer={},
        Logging={'save-dir': 'logs/debug'},
        # Measurements={},
    )
    plot_runs_svds(logs_parent_dir, run_config_params)
    plot_runs(logs_parent_dir, run_config_params)
    plot_runs_rel_trace(logs_parent_dir, run_config_params)


def _test():
    root_dir = '/home/marius/mit/research/NN_layerwise_analysis'
    log_dir = 'logs/'
    main(os.path.join(root_dir, log_dir))


if __name__ == '__main__':
    _test()
