import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

import os
import yaml

from Logger import SaveDirs
import utils

from typing import Dict, Any, Iterator
import warnings

FILETYPE = ".pdf"
# FILETYPE = ".png"

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
            savepath = os.path.join(savedir.plots, measure + FILETYPE)
            print(f"saving to {savepath}")
            plt.savefig(savepath)
            plt.show()


def plot_runs_svds(base_dir, run_config_params, selected_epochs=None):
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
            sub_selection = (measure_df['l_ord'] <= 24) & (measure_df['r_ord'].isin(['m']))  #  & (measure_df['l_type'].isin([-2]))

            layers = measure_df['layer_name'].unique()
            epochs = measure_df['epoch'].unique()

            savepath = os.path.join(savedir.plots, measure)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            savepath = os.path.join(savepath, r'e{epoch:0>3}/{layer}' + FILETYPE)
            print(f"saving to {savepath}")

            for epoch in tqdm.tqdm(epochs, leave=False):
                if selected_epochs is not None and epoch not in selected_epochs:
                    continue
                for layer in layers:
                    fig = plt.figure(figsize=(8, 6))
                    selection = (measure_df['epoch'] == epoch) & (measure_df['layer_name'] == layer) & sub_selection


                    corr_df = utils.corr_from_df(measure_df[selection])

                    sns.heatmap(
                        data=corr_df.applymap(abs),
                        cmap=sns.light_palette('red', as_cmap=True), vmin=0.0, vmax=0.8,
                        # data=corr_df,
                        # cmap='vlag', vmin=-0.8, vmax=0.8,
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


def plot_approx_rank(base_dir, run_config_params):
    """Plot the approximate rank of weight matrices."""

    relevant_measures = {
        'SingularValues': dict(x='layer_name', hue='epoch'),  # Remember to modify code too if needed when uncommenting others
    }

    for measure, plot_config in relevant_measures.items():
        # print(f"Plotting {measure}:")
        print(f"Plotting approximate rank")
        for run_dir in filter_configs(base_dir, run_config_params):
            print(f"\t{run_dir}", end=', ')
            savedir = SaveDirs(run_dir, timestamp_subdir=False, use_existing=True)

            fig = plt.figure(figsize=(10, 8))

            try:
                measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))
            except FileNotFoundError as e:
                warnings.warn(str(e))
                continue

            # Find first sigma where sum of singular values are 0.99 or greater
            approx_rank_cutoff = 0.99
            gt_cutoff_df = measure_df[(measure_df['value'] >= approx_rank_cutoff) & (measure_df['sum'].isin([True]))]  # Use only the sigma values greater than 99% (cumulative)
            approx_rank_df = gt_cutoff_df.loc[gt_cutoff_df.groupby(['epoch', 'layer_name'])['sigma_idx'].idxmin()]  # For each unique combination ['epoch', 'layer_name'], get the value with the lowest sigma value
            df = approx_rank_df
            del measure_df

            selection = (df['epoch'] != -1)
            # selection &= df['epoch'].isin([10, 50, 100, 200, 300])
            # selection &= (df['layer_name'].isin(['conv1', 'bn1', *[f'layer{i//2}.{i%2}' for i in range(2, 10)], 'avgpool', 'fc']))
            selection &= (df['layer_name'] != 'model')

            # Plot absolute traces
            ## If there is a hue-parameter and the x-axis has only one entry, replace the x-axis with the hue-parameter
            if 'hue' in plot_config.keys() and len(df[selection][plot_config['x']].unique()) == 1:
                plot_config['x'] = plot_config['hue']
                del plot_config['hue']

            # Do the plotting
            sns.lineplot(data=df[selection], y='sigma_idx', **plot_config)
            plt.title(f"Approximate rank (cutoff {approx_rank_cutoff}) over {plot_config['x']} for \n{os.path.split(savedir.base)[-1]}")

            plt.ylabel(r'Approximate rank')
            plt.yscale('log')

            plt.xticks(rotation=90)

            plt.tight_layout()
            savepath = os.path.join(savedir.plots, measure + "_ApproxRank" + FILETYPE)
            print(f"saving to {savepath}")
            plt.savefig(savepath)
            plt.show()



def plot_runs_rel_trace(base_dir, run_config_params, epoch=-1):
    """The base plotting function to copy and modify."""

    relevant_measures = {
        'SingularValues': dict(x='layer_name', hue='sigma_idx'),  # Remember to modify code too if needed when uncommenting others
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
            # selection = measure_df['epoch'].isin([10, 20, 50, 100, 200, 300]) & (measure_df['layer_name'] != 'model')
            selection = measure_df['epoch'].isin([10, 50, 100, 200, 300])
            # selection = (measure_df['epoch'] != -1)  # & (measure_df['layer_name'].isin(['conv1', 'bn1', *[f'layer{i//2}.{i%2}' for i in range(2, 10)], 'avgpool', 'fc']))
            selection &= (measure_df['layer_name'] != 'model')

            if measure == 'SingularValues':  # TODO(marius): Make less hacky
                if epoch == -1:
                    epoch = max(measure_df['epoch'])
                selection &= measure_df['sum'].isin([False])
                selection &= measure_df['sigma_idx'].isin([i for i in range(20)])
                selection &= measure_df['epoch'].isin([epoch])

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
            elif measure == 'SingularValues':
                plt.yscale('log')
                plt.ylabel(r'$\sum_{i=1}^{m}\sigma_i / \sum_{i}\sigma_i$')
                plt.title(f"Singular values as proportion of trace norm: Epoch {epoch}\n{os.path.split(savedir.base)[-1]}")

            plt.xticks(rotation=90)

            plt.tight_layout()
            savepath = os.path.join(savedir.plots, measure + FILETYPE)
            print(f"saving to {savepath}")
            plt.savefig(savepath)
            plt.show()


def abstract_plot():

    sns.set_theme(style='darkgrid')
    root_dir = '/home/marius/mit/research/NN_layerwise_analysis'
    savedir = SaveDirs(os.path.join(root_dir, 'logs/mlp_mnist'), timestamp_subdir=False, use_existing=True)

    # Do the trace plot:
    fig, axes = plt.subplots(2, 2, figsize=(12, 6)) #  , sharex='col')
    plt.sca(axes[0][0])
    measure = 'TraceMeasure'
    plot_config = dict(x='layer_name', hue='epoch', style='trace', )  # style_order=['sum', 'between', 'within'])
    measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))
    selection = measure_df['epoch'].isin([10, 100, 300]) & (measure_df['layer_name'] != 'model')

    # Replace stuff for formatting
    layer_replacing_dict = {f'model.block{i}.fc': str(i) for i in range(10)}
    layer_replacing_dict['model.fc'] = '10'
    measure_df = measure_df.replace({'layer_name': layer_replacing_dict})

    # sns.lineplot(data=measure_df[selection], y='value', **plot_config)
    # plt.title(f"{measure} over {plot_config['x']} for \n{os.path.split(savedir.base)[-1]}")
    # plt.yscale('log')

    # Plot relative trace
    if measure == "TraceMeasure":
        # plt.legend(loc='center left')
        plt.sca(axes[0][0])
        df = measure_df
        total_trace_sel = df['trace'] == 'sum'
        between_trace_sel = df['trace'] == 'between'
        within_trace_sel = df['trace'] == 'within'
        total_trace = df['value'].to_numpy()[total_trace_sel]
        df.at[between_trace_sel, 'value'] = df['value'].to_numpy()[between_trace_sel] / total_trace
        df.at[within_trace_sel, 'value'] = df['value'].to_numpy()[within_trace_sel] / total_trace
        sns.lineplot(data=df[selection][df[selection]['trace'] != 'sum'], y='value', **plot_config)
        # plt.yscale('log')

    # axes[0][0].get_legend().remove()
    axes[0][0].set_title("Metrics through neural network layers:\nRelative covariance contribution")
    # axes[0][0].set_ylabel("Trace")

    axes[0][0].legend(loc='best')
    axes[0][0].set_xlabel("Layer")
    axes[0][0].set_ylabel("Relative trace")

    # SVD plots:
    measure = 'MLPSVDMeasure'
    measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))
    sub_selection = (measure_df['l_ord'] <= 19) & (measure_df['r_ord'].isin(['m']))  # & (measure_df['l_type'].isin([-2]))

    epoch = 300
    layers = ['model.block4.fc', 'model.fc']

    for plot_idx, layer in enumerate(layers):
        plt.sca(axes[plot_idx][1])
        selection = (measure_df['epoch'] == epoch) & (measure_df['layer_name'] == layer) & sub_selection

        corr_df = utils.corr_from_df(measure_df[selection])

        # corr_df = corr_df.rename(index=[str(i) for i in range(10)], inplace=True)
        corr_df.index = pd.Index([f"{i+1}" for i in range(len(corr_df.index))])
        corr_df.columns = pd.Index([f"{i+1}" for i in range(len(corr_df.columns))])

        sns.heatmap(
            data=corr_df.applymap(abs), vmin=0.0, vmax=0.8,
            # cmap='mako_r', # vmin=0.0, vmax=1,
            cmap=sns.light_palette('red', as_cmap=True),
            # data=corr_df, vmin=-0.8, vmax=0.8,
            # cmap='vlag',
        )
        # plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.ylabel(
            # ("Layer 10:" if layer == 'model.fc' else "Layer 4:") +
            " Singular vector"
        )

    axes[0][1].set_title("Cosine angle between class means and singular vectors:\nLayer 4")
    axes[0][1].set_xlabel("Class mean")
    axes[1][1].set_title("Layer 10")
    axes[1][1].set_xlabel("Class mean")

    plt.tight_layout()
    plt.savefig('../tmp/trace_and_cosine' + FILETYPE)
    # plt.show()


    # Singular values plot:
    plt.sca(axes[1][0])
    measure = 'SingularValues'
    measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))
    measure_df = measure_df.replace({'layer_name': layer_replacing_dict})

    measure_df['sigma_idx'] = measure_df['sigma_idx'].map(lambda i: i+1)  # Make sigma 1-indexed
    # sigma_choice = [5, 10, 20, 40, 80]
    sigma_choice = [10]
    # sigma_choice = [i+1 for i in range(20)]

    selection = (measure_df['layer_name'] != 'model') & (measure_df['sum'].isin([True]))
    selection &= measure_df['sigma_idx'].isin(sigma_choice)
    selection &= measure_df['epoch'].isin([300])  # measure_df['sigma_idx'].isin([])

    sns.lineplot(
        data=measure_df[selection], y='value', x='layer_name',
        style='sigma_idx',
        markers=['s'],
        # hue='sigma_idx',
        # palette=sns.color_palette("husl", len(sigma_choice)),
        # hue='epoch',
    )

    # plt.ylabel("Relative sum")
    # plt.title(f"{measure} over {plot_config['x']} for \n{os.path.split(savedir.base)[-1]}")
    plt.xlabel("Layer")
    # plt.ylabel(r'$\sum_{i=1}^{10}\sigma_i / \sum_{i}\sigma_i$')
    plt.ylabel('Normalized cumulative sum')
    # plt.title(f"First n singular vectors part of total variance for \n{os.path.split(savedir.base)[-1]}")
    plt.title('Sum of first $C=10$ singular values as proportion of total sum')
    axes[1][0].get_legend().remove()
    # plt.legend(title=None, labels=[r'$\dfrac{\sum^{10}_{i=1} \sigma_i}{\sum_{i=0}^{512}\sigma_{i}}$'])
    # plt.legend(title=None, labels=[''])

    plt.tight_layout()
    # plt.savefig('../tmp/singular_values' + FILETYPE)
    plt.savefig('../tmp/trace_singularvalues_cosine' + FILETYPE)
    plt.show()

    pass


def main(logs_parent_dir: str):
    """Run some standard plotting on the measurements. Prone to failure!!!"""
    sns.set_theme(style='darkgrid')
    run_config_params = dict(  # All parameters must match what is given here.
        # Model={'model-name': 'resnet18'},
        # Data={'dataset-id': 'cifar10'},
        # Optimizer={},
        # Logging={'save-dir': 'logs/mlp_wide_nobias_nobn_mnist'},
        # Measurements={},
    )
    plot_approx_rank(logs_parent_dir, run_config_params)
    # plot_runs_svds(logs_parent_dir, run_config_params, selected_epochs=[300])
    # plot_runs(logs_parent_dir, run_config_params)
    # plot_runs_rel_trace(logs_parent_dir, run_config_params)
    # for i in [0, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 250, 300]:
    #     plot_runs_rel_trace(logs_parent_dir, run_config_params, epoch=i)


def _test():
    root_dir = '/home/marius/mit/research/NN_layerwise_analysis'
    log_dir = 'logs/'
    main(os.path.join(root_dir, log_dir))


if __name__ == '__main__':
    _test()
    # abstract_plot()
