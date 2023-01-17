import matplotlib.legend
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

import os
import yaml

from Logger import SaveDirs
import utils

from typing import Dict, Any, Iterator, Tuple, Optional, Union
import warnings

# FILETYPE = ".pdf"
FILETYPE = ".png"

PRETTY_OUT = True
FIGSIZE_BASE = 2  # 2

class plot_utils:
    @staticmethod
    def filter_configs(base_dir: str, required_params: Dict[str, Dict[str, Any]], recurse: bool = True) -> Iterator[str]:
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
                    for child_dir in plot_utils.filter_configs(base_dir=os.path.join(base_dir, run_dirname), required_params=required_params, recurse=recurse):
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

    @staticmethod
    def add_nc_line(df: pd.DataFrame, nc_layer_name: Optional[str]):
        """Plot vertical line when NC happens"""
        if nc_layer_name is not None:
            try:
                layer_x_coord = df['layer_name'].cat.categories.get_loc(nc_layer_name)
            except KeyError:
                return
            plt.axvline(layer_x_coord,
                        color='g', linestyle='--',
                        # label="Neural Collapse"
                        )
            # plt.legend()

    @classmethod
    def get_nc_layer(cls, savedir: SaveDirs) -> Optional[Union[int, str]]:
        """Get the first layer of neural collapse for the last epoch in the specified run"""
        print("\tFinding NC epoch: ", end='')

        # Condition for Neural Collapse:
        measure, condition = 'CDNV', lambda df: df['value'] < 0.2  # 0.1
        # measure, condition = 'Traces', lambda df:  (df['trace'] == 'within') & (df['value'] < 0.1)

        try:
            measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))
        except FileNotFoundError as e:
            print(f"\tError, no file {os.path.join(savedir.measurements, measure + '.csv')}")
            return None

        layer_order = measure_df['layer_name'].unique()
        measure_df['layer_name'] = measure_df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        last_epoch = measure_df['epoch'].max()
        if last_epoch != 300:
            warnings.warn(f"Last epoch is not 300 but {last_epoch}. Make sure this does not break the plot.")
        # layer_map = {layer_name: idx for idx, layer_name in enumerate(measure_df['layer_name'].unique())}

        satisfies_collapse = condition(measure_df)
        first_clp = measure_df[satisfies_collapse].groupby(['epoch'])['layer_name'].min()

        return first_clp.get(last_epoch, default=None)

    @staticmethod
    def reformat_layer_names(layer_names: pd.Series) -> pd.Series:
        """Reformat layer names to make them more presentable in plots"""
        assert hasattr(layer_names, 'cat'), "Pandas series layer_names is not categorical!"

        warnings.warn("TODO: Implement layer name mapper")  # TODO(marius)

        return layer_names

    @staticmethod
    def capitalize_legend(legend: matplotlib.legend.Legend):
        return
        def make_nice(x: plt.Text):
            t = x.get_text().replace('_', ' ')
            if t:
                t = t[0].upper() + t[1:]
            return x.set_text(t)

        legend.set_title(make_nice(legend.get_title()))
        # for line in

    @staticmethod
    def get_xticks(layer_names: pd.Series):
        return layer_names.cat.codes.unique()


class NCPlotter:
    standard_epochs = [10, 100, 300]
    # standard_epochs = set(range(0, 601))

    @classmethod
    def plot_runs(cls, base_dir, run_config_params):
        """The base plotting function to copy and modify."""

        for run_dir in plot_utils.filter_configs(base_dir, run_config_params):
            print(f"\nPlotting {run_dir}:")
            savedir = SaveDirs(run_dir, timestamp_subdir=False, use_existing=True)
            # fix, axes = plt.subplots(nrows=None, ncols=None, sharex='all')
            nc_layer = plot_utils.get_nc_layer(savedir)
            print("Layer of NC after max epochs: ", nc_layer)

            # Plot ax(es) for each measure
            for measure, (plot_func, num_axes) in cls.get_relevant_measures().items():
                # fig = plt.figure(figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
                print(f"\t{measure}", end=', ')
                try:
                    measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))
                except FileNotFoundError as e:
                    print(f"\tError, no file {os.path.join(savedir.measurements, measure + '.csv')}")
                    # warnings.warn(str(e))
                    continue

                # super_selection = measure_df['epoch'].isin([10, 20, 50, 100, 200, 300])
                ret = plot_func(measure_df, nc_layer=nc_layer)
                if ret is None:  # If plotting fails, don't save it
                    warnings.warn("Plotting failed")
                    continue

                # plt.suptitle(f"{measure} for \n{os.path.relpath(savedir.base, savedir.root_dir)}")
                # Add Dataset to title
                if 'cifar10' in savedir.base.lower():
                    title = 'CIFAR10'
                elif 'fashionmnist' in savedir.base.lower():
                    title = 'FashionMNIST'
                elif 'mnist' in savedir.base.lower():
                    title = 'MNIST'
                elif 'svhn' in savedir.base.lower():
                    title = 'SVNH'
                else:
                    title = 'OtherData'
                # Add model to title
                if 'mlp' in savedir.base.lower():
                    title = "MLP: " + title
                elif 'convnet_deep' in savedir.base.lower():
                    title = "Convnet: " + title
                elif 'convnet_huge' in savedir.base.lower():
                    title = "Convnet huge: " + title
                elif 'resnet18' in savedir.base.lower():
                    title = "Resnet18: " + title
                elif 'vgg16_bn' in savedir.base.lower():
                    title = "VGG16: " + title
                else:
                    title = "OtherModel: " + title

                plt.suptitle(title, y=0.99)
                if PRETTY_OUT:
                    # plt.gca().tick_params(axis='x', labelsize=10)
                    ax = plt.gca()
                    xticks = ax.get_xticklabels()
                    new_xticks = []
                    for idx, tickname in enumerate(xticks):
                        new_xticks.append(tickname if (idx+1) % 2 else '')
                    if len(xticks) > 15 and measure != 'Accuracy':
                        ax.set_xticks(ax.get_xticks())
                        ax.set_xticklabels(new_xticks)
                plt.tight_layout(pad=0.2)
                savepath = os.path.join(savedir.plots, measure + FILETYPE)
                print(f"saving to {savepath}")
                plt.savefig(savepath)
                # plt.show()
                plt.close()

    @staticmethod
    def _plot_accuracy(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
            axes = (ax,)
        plt.sca(axes[0])
        sns.lineplot(data=df, x='epoch', y='value', hue='split')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plot_utils.capitalize_legend(plt.gca().get_legend())

        plt.title('Accuracy')

        return axes

    @staticmethod
    def _plot_traces(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes, plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            # plt.figure()
            fig, axes = plt.subplots(2, 1, sharex='all', figsize=(3*FIGSIZE_BASE, 2.5*FIGSIZE_BASE))
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        selection = df['epoch'].isin(NCPlotter.standard_epochs)  # & (measure_df['layer_name'] != 'model') selection &= measure_df['epoch'].isin([10, 50, 100, 200, 300])
        selection &= df['layer_name'] != 'model'

        # if 'hue' in plot_config.keys() and len(measure_df[selection][plot_config['x']].unique()) == 1:
        #     plot_config['x'] = plot_config['hue']
        #     del plot_config['hue']
        plot_config = dict(x='layer_name', hue='epoch', style='trace', style_order=['sum', 'between', 'within'])

        plot_utils.add_nc_line(df, nc_layer)
        sns.lineplot(data=df[selection], y='value', **plot_config)

        plt.yscale('log')
        plt.legend(loc='center left')

        plot_utils.capitalize_legend(plt.gca().get_legend())
        plt.ylabel('Trace')

        plt.title('Trace of covariances: Within- and between-class')
        plt.sca(axes[1])
        total_trace_sel = df['trace'] == 'sum'
        between_trace_sel = df['trace'] == 'between'
        within_trace_sel = df['trace'] == 'within'
        total_trace = df['value'].to_numpy()[total_trace_sel]
        df.at[between_trace_sel, 'value'] = df['value'].to_numpy()[between_trace_sel] / total_trace
        df.at[within_trace_sel, 'value'] = df['value'].to_numpy()[within_trace_sel] / total_trace

        plot_utils.add_nc_line(df, nc_layer)
        sns.lineplot(data=df[selection][df[selection]['trace'] != 'sum'], y='value', **plot_config)
        plt.yscale('log')
        plt.legend(loc='center left')
        plt.xticks(rotation=90)

        plot_utils.capitalize_legend(plt.gca().get_legend())
        plt.xlabel('Layer')
        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
            plt.xticks(rotation=0)
        plt.ylabel('Relative Trace')

        axes[1].get_legend().remove()

        return axes

    @staticmethod
    def _plot_ETF(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes, plt.Axes, plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, axes = plt.subplots(3, 1, sharex='all', figsize=(3*FIGSIZE_BASE, 3*FIGSIZE_BASE))
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        plot_config = dict(x='layer_name', hue='epoch')

        selection = df['epoch'].isin(NCPlotter.standard_epochs)
        selection &= df['layer_name'] != 'model'
        sel_df = df[selection]

        mean_df = sel_df.groupby(['epoch', 'layer_name', 'type'], as_index=False).mean().reset_index()
        std_df = sel_df.groupby(['epoch', 'layer_name', 'type'], as_index=False).std().reset_index()

        # Sort layer names to ensure they are in the correct order in the plot
        layer_order = {layer_name: idx for idx, layer_name in enumerate(df['layer_name'].unique())}
        mean_df.sort_values(by='layer_name', key=lambda layer_names: layer_names.map(layer_order), inplace=True)
        std_df.sort_values(by='layer_name', key=lambda layer_names: layer_names.map(layer_order), inplace=True)

        # print("Doing 1/(C-1) correction", end=", ")
        # mean_df['value'].loc[std_df['type'] == 'angle'] = mean_df[std_df['type'] == 'angle']['value'] + 1/(10-1)
        plot_utils.add_nc_line(mean_df, nc_layer)
        sns.lineplot(data=mean_df[mean_df['type'] == 'angle'], y='value', **plot_config)
        plt.title(r'Mean of $1/(1-C) + \cos(\mu_i, \mu_j)$')

        plot_utils.capitalize_legend(plt.gca().get_legend())
        # plt.ylabel(r'Avg $\frac{1}{1-C} + \cos(\mu_i, \mu_j)$')
        plt.ylabel(None)

        ###
        plt.sca(axes[1])
        plot_utils.add_nc_line(mean_df, nc_layer)
        sns.lineplot(data=std_df[std_df['type'] == 'angle'], y='value', **plot_config)
        plt.title(r'Std of $1/(1-C) + \cos(\mu_i, \mu_j)$')

        plot_utils.capitalize_legend(plt.gca().get_legend())
        # plt.ylabel(r'Std $\frac{1}{1-C} + \cos(\mu_i, \mu_j)$')
        plt.ylabel(r'Std of $\frac{1}{1-C} + \cos(\mu_i, \mu_j)$')
        plt.ylabel(None)

        plt.sca(axes[2])
        plot_utils.add_nc_line(mean_df, nc_layer)
        rel_std_df = std_df.copy()
        rel_std_df['value'] /= mean_df['value']
        sns.lineplot(data=rel_std_df[std_df['type'] == 'norm'], y='value', **plot_config)
        plt.title(r'Relative std of $\|\mu_i\|_2$')

        plot_utils.capitalize_legend(plt.gca().get_legend())
        # plt.ylabel(r'Std $\|\mu_i\|_2$ / Avg $\|\mu_i\|_2$')
        plt.ylabel(None)

        plt.xlabel('Layer')
        plt.xticks(rotation=30)
        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
            plt.xticks(rotation=0)

        axes[0].get_legend().remove()
        axes[2].get_legend().remove()

        return axes

    @staticmethod
    def _plot_weightSVs(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
            axes = (ax,)
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        plot_utils.add_nc_line(df, nc_layer)

        max_sv = 30

        epoch = max(df['epoch'])
        selection = df['sum'].isin([False])
        # selection &= df['sigma_idx'].isin([i for i in range(max_sv)])
        selection &= df['epoch'].isin([epoch])
        selection &= df['layer_name'] != 'model'

        df_sel = df[selection]
        # sns.lineplot(data=df[selection], x='layer_name', y='value', hue='sigma_idx')

        sv_first_10 = df_sel['sigma_idx'].isin([i for i in range(10)])
        sv_after_10 = df_sel['sigma_idx'].isin([i for i in range(10, max_sv)])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_sel.loc[:, ('sigma_idx',)] = df_sel['sigma_idx'].map(lambda x: x+1)  # Make sigmas 1-index in presentation

        sns.lineplot(data=df_sel[sv_first_10], x='layer_name', y='value', hue='sigma_idx', palette='dark:red')
        sns.lineplot(data=df_sel[sv_after_10], x='layer_name', y='value', hue='sigma_idx', palette='dark:#ADF', legend='brief')
        # plt.legend(title='Sing. val. idx', labels=['First 10', f'11-{max_sv}'])   # TODO(marius): Make legends
        # sns.lineplot(data=df_sel[sv_first_10], x='layer_name', y='value', label='First 10', color='red', ci=100)


        plt.yscale('log')
        # plt.ylabel(r'$\sum_{i=1}^{m}\sigma_i / \sum_{i}\sigma_i$')

        plot_utils.capitalize_legend(plt.gca().get_legend())
        # plt.ylabel(r'$\sigma_i / \sum_{j}\sigma_j$')
        plt.ylabel(r'$\sigma_i / \sum\sigma_j$')
        plt.xlabel('Layer')

        plt.title(f"Weight relative SVs, epoch {epoch}")
        plt.xticks(rotation=90)

        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
            plt.xticks(rotation=0)

        lgd = axes[0].get_legend()
        # mask, labels = [0, 3, 6, 7, 9, 11], [0, 5, 10, 11, 20, 30]
        mask, labels = [6, 11], [' 1-10', '11-30']
        handles = [line for idx, line in enumerate(lgd.legendHandles) if idx in mask]
        # texts = [plt.Text(0, 0, str(i)) for i in [0, 10, 11, 30]]
        plt.legend(handles=handles, labels=labels, title="SV idx $i$")

        # axes[0].get_legend().remove()

        return axes

    @staticmethod
    def _plot_angleBetweenSubspaces(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
            axes = (ax,)
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        if 'sigma_idx' in df.columns:
            warnings.warn("Old evaluation of angleBetweenSubspace, skipping.")
            return None

        eval_rank = 10

        selection = df['epoch'].isin([0, 300])
        # selection = df['epoch'].isin([0, 1, 2, 3])
        # selection = df['epoch'].isin(NCPlotter.standard_epochs)
        selection &= df['layer_name'] != 'model'
        selection &= df['rank'] == eval_rank

        # Divide by number of singular values (for experiments post 2022-10-28)
        # df['value'] = df['value'].map(lambda val: val / num_sing_vals)  # TODO(marius): Remove "*10" and update AngleBetweenSubspaces measurer (i.e. remove "/10")

        plot_utils.add_nc_line(df, nc_layer)
        sns.lineplot(data=df[selection], x='layer_name', y='value', hue='epoch',
                     palette=['r', 'k'] if len(df[selection]['epoch'].unique()) == 2 else None,
                     )

        plot_utils.capitalize_legend(plt.gca().get_legend())
        plt.ylabel(r'PABS')
        plt.xlabel('Layer')

        plt.title(f"Angle between subspaces, rank {eval_rank}")
        plt.yscale('linear')
        plt.ylim([None, 1.04])
        plt.xticks(rotation=90)

        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
                plt.gca().set_xticklabels(plot_utils.get_xticks(df['layer_name']))
            plt.xticks(rotation=0)

        return axes

    @staticmethod
    def _plot_NCC(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
            axes = (ax,)
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        # selection = df['epoch'].isin(NCPlotter.standard_epochs)
        max_epoch = df['epoch'].max()
        selection = df['epoch'].isin([max_epoch])
        selection &= df['layer_name'] != 'model'

        plot_utils.add_nc_line(df, nc_layer)
        # sns.lineplot(data=df[selection], x='layer_name', y='value', hue='epoch', style='split')
        sns.lineplot(data=df[selection], x='layer_name', y='value', style='split')

        plot_utils.capitalize_legend(plt.gca().get_legend())
        plt.ylabel(r'NCC Accuracy')
        plt.xlabel('Layer')

        # plt.title(f"Accuracy of nearest-class-center classifier (NC4) for epoch {max_epoch}")
        plt.title(f"NCC Accuracy, epoch {max_epoch}")
        plt.xticks(rotation=90)
        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
            plt.xticks(rotation=0)


        return axes

    @staticmethod
    def _plot_cdnv(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
            axes = (ax,)
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        selection = df['epoch'].isin(NCPlotter.standard_epochs)
        selection &= df['layer_name'] != 'model'

        # print("Doing 1/(C^2-C) correction", end=", ")
        # df['value'].loc[selection] = df['value'].loc[selection] / (10 * (10-1))

        plot_utils.add_nc_line(df, nc_layer)
        sns.lineplot(data=df[selection], x='layer_name', y='value', hue='epoch')

        plot_utils.capitalize_legend(plt.gca().get_legend())
        plt.ylabel(r'CDNV')
        plt.xlabel('Layer')

        plt.title(f'CDNV')
        plt.yscale('log')
        plt.xticks(rotation=90)

        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
            plt.xticks(rotation=0)

        return axes

    @staticmethod
    def _plot_NC1(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
            axes = (ax,)
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        selection = df['epoch'].isin(NCPlotter.standard_epochs)
        selection &= df['layer_name'] != 'model'

        plot_utils.add_nc_line(df, nc_layer)
        sns.lineplot(data=df[selection], x='layer_name', y='value', hue='epoch')

        plot_utils.capitalize_legend(plt.gca().get_legend())
        plt.ylabel(r'$\Sigma_B^\dagger \Sigma_W$')
        plt.xlabel('Layer')

        plt.title(f'NC1')
        plt.yscale('log')
        plt.xticks(rotation=90)

        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
            plt.xticks(rotation=0)

        return axes

    @staticmethod
    def _plot_activationCovSVs(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
            axes = (ax,)
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        selection = df['epoch'].isin(NCPlotter.standard_epochs)
        selection &= df['layer_name'] != 'model'

        # subselection &= df['class_idx'].isin([0, 1])  # For when using cifar10_doubleclass

        sel_df = df[selection]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # sel_df['value_inv'] = sel_df[sel_df['type'] == 'within_single']['value'].apply(lambda val: 1/val)
            sel_df['value_sq_normed_inv'] = (
                sel_df[sel_df['type'] == 'within_single_absolute']
                .groupby(['epoch', 'layer_name', 'class_idx'])['value']
                .transform(lambda val: (val**2).sum()/val**2)
            )  # For sum_i sigma_i^2 / sigma_max^2

        class_largest_sv_df = sel_df[sel_df['sigma_idx'].isin([0]) & sel_df['sum'].isin([False])]

        plot_utils.add_nc_line(df, nc_layer)
        sns.lineplot(data=class_largest_sv_df, x='layer_name', y='value_sq_normed_inv', hue='epoch',
                     ci=None,
                     )

        plot_utils.capitalize_legend(plt.gca().get_legend())
        plt.ylabel('Stable rank')
        plt.xlabel('Layer')

        plt.title(f"Within class covariance stable rank")
        # plt.yscale('log')
        plt.xticks(rotation=90)

        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
            plt.xticks(rotation=0)

        return axes

    '''For plotting the distribution of SVs
    @staticmethod
    def _plot_activationCovSVs(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
            axes = (ax,)
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        # df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        # selection = df['epoch'].isin([0, 1] + NCPlotter.standard_epochs)
        selection = df['epoch'].isin([0, 10, 300])
        # selection &= df['layer_name'] != 'model'
        # selection &= df['layer_name'].isin(['model.block3.fc', 'model.block8.fc'])
        selection &= df['layer_name'].isin(['model.block3.fc', 'model.block8.fc'])

        subselection = df['type'] == 'within_single'
        subselection &= df['sum'].isin([False])

        sel_df = df[selection & subselection]
        # grouped_df = sel_df.groupby(['epoch', 'layer_name', 'class_idx'], as_index=False)
        class_largest_sv_df = sel_df[sel_df['sigma_idx'].isin([0]) & sel_df['sum'].isin([False])]
        # class_sum_sv_df = sel_df[sel_df['sigma_idx'] == sel_df['sigma_idx'].max()][sel_df['sum'].isin([True])]  # Is always 1

        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        #     class_largest_sv_df.loc[:, ('value',)] = class_largest_sv_df['value'].apply(lambda v: 1/v)

        # plot_utils.add_nc_line(df, nc_layer)
        # sns.lineplot(data=class_largest_sv_df, x='layer_name', y='value', hue='epoch',
        #              ci=None,
        #              )
        sns.lineplot(data=sel_df, x='sigma_idx', y='value', hue='epoch',
                     style='layer_name',
                     ci=None,
                     palette=['r', 'p', 'g'],
                     )

        plot_utils.capitalize_legend(plt.gca().get_legend())
        # plt.ylabel('Stable rank')
        # plt.xlabel('Layer')
        plt.xlabel("SV idx.")
        plt.ylabel("Rel sing. val.")
        plt.yscale('log')
        # plt.ylim([10**(-6), 0.2])


        # plt.title(f"Within class covariance stable rank")
        # plt.yscale('log')
        plt.xticks(rotation=90)

        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
            plt.xticks(rotation=0)

        return axes

    '''

    @staticmethod
    def _plot_activationStableRank(df: pd.DataFrame, axes: Optional[Tuple[plt.Axes]] = None, nc_layer: Optional[str] = None):
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(2*FIGSIZE_BASE, 2*FIGSIZE_BASE))
            axes = (ax,)
        plt.sca(axes[0])
        layer_order = df['layer_name'].unique()
        df['layer_name'] = df['layer_name'].astype(pd.api.types.CategoricalDtype()).cat.set_categories(layer_order, ordered=True)

        selection = df['epoch'].isin(NCPlotter.standard_epochs)
        selection &= df['layer_name'] != 'model'

        plot_utils.add_nc_line(df, nc_layer)
        sns.lineplot(data=df[selection], x='layer_name', y='value', hue='epoch',
                     ci=None,
                     )

        plot_utils.capitalize_legend(plt.gca().get_legend())
        plt.ylabel('Square Stable rank of H')
        plt.xlabel('Layer')

        plt.title(f"Square Stable Rank of activation matrix H")
        # plt.yscale('log')
        plt.xticks(rotation=90)

        if PRETTY_OUT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.gca().set_xticklabels(plot_utils.get_xticks(df[selection]['layer_name']))
            plt.xticks(rotation=0)

        return axes

    @staticmethod
    def get_relevant_measures() -> Dict[str, Tuple[callable, int]]:
        relevant_measures = {
            # Other:
            'Accuracy': (NCPlotter._plot_accuracy, 1),
            'CDNV': (NCPlotter._plot_cdnv, 1),
            'ActivationCovSVs': (NCPlotter._plot_activationCovSVs, 1),
            'ActivationStableRank': (NCPlotter._plot_activationStableRank, 1),
            # NC1:
            'NC1': (NCPlotter._plot_NC1, 1),
            'Traces': (NCPlotter._plot_traces, 2),
            # NC2:
            'ETF': (NCPlotter._plot_ETF, 3),
            'WeightSVs': (NCPlotter._plot_weightSVs, 1),
            # NC3:
            'AngleBetweenSubspaces': (NCPlotter._plot_angleBetweenSubspaces, 1),
            # NC4:
            'NCC': (NCPlotter._plot_NCC, 1)
        }
        return relevant_measures

    pass


'''
def plot_runs_svds(base_dir, run_config_params, selected_epochs=None):
    """For plotting the SVD correlation matrices."""

    relevant_measures = {
        'MLPSVD': None  # dict(x='epoch', hue='split', style=None)
    }

    for measure, plot_config in relevant_measures.items():
        print(f"Plotting {measure}:")
        for run_dir in plot_utils.filter_configs(base_dir, run_config_params):
            print(f"\t{run_dir}", end=', ')
            savedir = SaveDirs(run_dir, timestamp_subdir=False, use_existing=True)
            # TODO(marius): Merge dataframes and plot
            try:
                measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))
            except FileNotFoundError as e:
                warnings.warn(str(e))
                continue
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
        for run_dir in plot_utils.filter_configs(base_dir, run_config_params):
            print(f"\t{run_dir}", end=', ')
            savedir = SaveDirs(run_dir, timestamp_subdir=False, use_existing=True)


            try:
                measure_df = pd.read_csv(os.path.join(savedir.measurements, measure + '.csv'))
            except FileNotFoundError as e:
                warnings.warn(str(e))
                continue

            # fig = plt.figure(figsize=(10, 2*8))
            fix, axes = plt.subplots(3, 1, sharex='all', figsize=(10, 2*8))
            """
            ### Do Approx rank plot:
            # Find first sigma where sum of singular values are 0.99 or greater
            approx_rank_cutoff = 0.99
            gt_cutoff_df = measure_df[(measure_df['value'] >= approx_rank_cutoff) & (measure_df['sum'].isin([True]))]  # Use only the sigma values greater than 99% (cumulative)
            approx_rank_df = gt_cutoff_df.loc[gt_cutoff_df.groupby(['epoch', 'layer_name'])['sigma_idx'].idxmin()]  # For each unique combination ['epoch', 'layer_name'], get the value with the lowest sigma value
            df = approx_rank_df
            # del measure_df

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
            """
            ### Do sing_val_cutoff plot:
            # fig = plt.figure(figsize=(10, 8))
            plt.sca(axes[0])

            # Approx rank
            cutoffs = [0.9, 0.97, 0.99]

            cutoff_df_list = []
            for cutoff in cutoffs:
                cutoff_df = measure_df[(measure_df['value'] > cutoff) & (measure_df['sum'].isin([True]))]  # Use only the sigma values greater than 99% (cumulative)
                cutoff_df = cutoff_df.loc[cutoff_df.groupby(['epoch', 'layer_name'])['sigma_idx'].idxmin()]  # For each unique combination ['epoch', 'layer_name'], get the value with the lowest sigma value
                cutoff_df.insert(len(cutoff_df.columns), 'cutoff', cutoff)
                cutoff_df_list.append(cutoff_df)

            df = pd.concat(cutoff_df_list, ignore_index=True)

            selection = df['epoch'] == df['epoch'].max()
            # selection = (df['epoch'] != -1)
            # selection &= df['epoch'].isin([10, 50, 100, 200, 300])
            # selection &= (df['layer_name'].isin(['conv1', 'bn1', *[f'layer{i//2}.{i%2}' for i in range(2, 10)], 'avgpool', 'fc']))
            selection &= (df['layer_name'] != 'model')

            if 'hue' in plot_config.keys() and len(df[selection][plot_config['x']].unique()) == 1:
                plot_config['x'] = plot_config['hue']
                del plot_config['hue']

            # Do the plotting
            sns.lineplot(data=df[selection], y='sigma_idx', x=plot_config['x'], hue='cutoff', palette='tab10',)  # **plot_config)
            plt.title(f"Approximate rank over {plot_config['x']} for \n{os.path.split(savedir.base)[-1]}")

            plt.ylabel(r'Approximate rank')
            plt.yscale('log')

            plt.xticks(rotation=90)

            # plt.tight_layout()
            # savepath = os.path.join(savedir.plots, measure + "_ApproxRankCutoffs" + FILETYPE)
            # print(f"saving to {savepath}")
            # plt.savefig(savepath)
            # plt.show()


            ### Do Stable Rank plot:

            # fig = plt.figure(figsize=(10, 8))
            plt.sca(axes[1])

            stable_rank_sel = (measure_df['sum'].isin([True])) & (measure_df['sigma_idx'] == 0)
            df = measure_df[stable_rank_sel]
            df['value'] = 1 / (df['value'].to_numpy() + 1E-24)

            selection = (df['epoch'] != -1)
            # selection &= df['epoch'].isin([10, 50, 100, 200, 300])
            # selection &= (df['layer_name'].isin(['conv1', 'bn1', *[f'layer{i//2}.{i%2}' for i in range(2, 10)], 'avgpool', 'fc']))
            selection &= (df['layer_name'] != 'model')

            if 'hue' in plot_config.keys() and len(df[selection][plot_config['x']].unique()) == 1:
                plot_config['x'] = plot_config['hue']
                del plot_config['hue']

            # Do the plotting
            sns.lineplot(data=df[selection], y='value', **plot_config)
            plt.title(f"Stable rank over {plot_config['x']} for \n{os.path.split(savedir.base)[-1]}")

            plt.ylabel(r'Stable rank $\|A\|_F \;/\; \|A\|_2$')  # == $\sum_{i}\sigma_i \; / \max{\sigma_i}$
            plt.yscale('log')

            plt.xticks(rotation=90)

            # plt.tight_layout()
            # savepath = os.path.join(savedir.plots, measure + "_StableRank" + FILETYPE)
            # print(f"saving to {savepath}")
            # plt.savefig(savepath)
            # plt.show()


            ### Do sing_val_cutoff plot:
            # fig = plt.figure(figsize=(10, 8))
            plt.sca(axes[2])

            cutoffs = [3E-2, 1E-2, 3E-3, 1E-3, 3E-4, 1E-4]

            cutoff_df_list = []
            for cutoff in cutoffs:
                cutoff_df = measure_df[(measure_df['value'] < cutoff) & (measure_df['sum'].isin([False]))]  # Use only the sigma values greater than 99% (cumulative)
                cutoff_df = cutoff_df.loc[cutoff_df.groupby(['epoch', 'layer_name'])['sigma_idx'].idxmin()]  # For each unique combination ['epoch', 'layer_name'], get the value with the lowest sigma value
                cutoff_df.insert(len(cutoff_df.columns), 'cutoff', cutoff)
                cutoff_df_list.append(cutoff_df)

            df = pd.concat(cutoff_df_list, ignore_index=True)

            selection = df['epoch'] == df['epoch'].max()
            # selection = (df['epoch'] != -1)
            # selection &= df['epoch'].isin([10, 50, 100, 200, 300])
            # selection &= (df['layer_name'].isin(['conv1', 'bn1', *[f'layer{i//2}.{i%2}' for i in range(2, 10)], 'avgpool', 'fc']))
            selection &= (df['layer_name'] != 'model')

            if 'hue' in plot_config.keys() and len(df[selection][plot_config['x']].unique()) == 1:
                plot_config['x'] = plot_config['hue']
                del plot_config['hue']

            # Do the plotting
            sns.lineplot(data=df[selection], y='sigma_idx', x=plot_config['x'], hue='cutoff', palette='tab10',
                         )  # **plot_config)
            plt.title(f"Cutoff rank over {plot_config['x']} after {df['epoch'].max()} epochs for \n{os.path.split(savedir.base)[-1]}")

            plt.ylabel(r'Sigmas larger than cutoff: $\min_i$ s.t. $(\frac{\sigma_i}{\sum_i \sigma_i} < $ cutoff)')  # == $\sum_{i}\sigma_i \; / \max{\sigma_i}$
            plt.yscale('log')

            plt.xticks(rotation=90)

            plt.tight_layout()
            # savepath = os.path.join(savedir.plots, measure + "_CutoffRank" + FILETYPE)
            savepath = os.path.join(savedir.plots, measure + "_Ranks" + FILETYPE)
            print(f"saving to {savepath}")
            plt.savefig(savepath)
            # plt.savefig(f'../tmp/{os.path.split(savedir.base)[-1]}_{measure}_ranks' + FILETYPE)
            plt.show()
'''

def main(logs_parent_dir: str):
    """Run some standard plotting on the measurements. Prone to failure!!!"""
    sns.set_theme(style='darkgrid')
    run_config_params = dict(  # All parameters must match what is given here.
        # Model={'model-name': 'convnet_deep'},
        # Data={'dataset-id': 'cifar10'},
        # Optimizer={},
        # Logging={'save-dir': 'logs/mlp_sharedweight_xwide_nobn_mnist'},
        # Logging={'save-dir': 'logs/debug'}
        # Measurements={},
    )
    NCPlotter.plot_runs(logs_parent_dir, run_config_params)
    # plot_approx_rank(logs_parent_dir, run_config_params)
    # plot_runs_svds(logs_parent_dir, run_config_params, selected_epochs=[300])
    # plot_runs(logs_parent_dir, run_config_params)
    # plot_runs_rel_trace(logs_parent_dir, run_config_params)
    # for i in [0, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 250, 300]:
    #     plot_runs_rel_trace(logs_parent_dir, run_config_params, epoch=i)


def _test():
    # root_dir = '/home/marius/mit/research/NN_layerwise_analysis'
    save_dir = SaveDirs('logs', timestamp_subdir=False, use_existing=True)

    log_dirs = []
    # log_dirs.append('')
    # log_dirs.append('matrix/convnet/2022-10-24T17:20/convnet_deep/cifar10/lr_0.01')
    # log_dirs.append('matrix/2022-10-11T20:21/mlp')
    log_dirs.append('matrix/2022-11-03T20:02/')

    for log_dir in log_dirs:
        main(os.path.join(save_dir.base, log_dir))


if __name__ == '__main__':
    _test()
    # abstract_plot()
