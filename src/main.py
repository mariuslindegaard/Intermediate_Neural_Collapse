import argparse
import warnings

from Experiment import Experiment
from Plotting import NCPlotter
import slurm_utils
import Logger
import os


def run_experiment(config_file_path: str, plot: bool, do_measurements: bool = True):
    """Run the experiment corresponding to the config file"""
    # TODO(marius): Add option to clean before training (i.e. removing directory before run)
    # TODO(marius): Add option to throw error if measurements already exist. (Maybe lower level code?)
    # TODO(marius): Add plotting automatically
    print("Loading experiment")
    exp = Experiment(config_file_path)
    print('NOT TRAINING!!!')  # TODO(marius): Remove debug
    # print("Training NN")
    # exp.train()

    # do_measurements = False  # TODO(marius): Remove debug
    if do_measurements:
        print("Running measurements")
        exp.do_measurements_on_checkpoints()
    if plot:
        print('Plotting results')
        NCPlotter.plot_runs(exp.logger.save_dirs.base, dict())



def main(config_file_path: str, unpack_config_matrix: bool, parse_and_submit_to_slurm: bool,
         use_timestamp_with_matrix: bool, dry_run: bool, plot_after_run: bool, no_measurements: bool):
    if unpack_config_matrix:
        print(f"Parsing matrix config at {config_file_path} and submitting to slurm.")
        configs_with_path, parent_savedir = slurm_utils.parse_config_matrix(config_file_path)
        if dry_run:
            print(f'\nDoing dry-run: Not submitting to slurm nor creating savedirs: {len(configs_with_path)} jobs.')
            return
        base_savedir = Logger.SaveDirs(parent_savedir, timestamp_subdir=use_timestamp_with_matrix)

        # Create directory, config file and run-script for each of the configs
        for idx, (config_dict, rel_savedir) in enumerate(configs_with_path):
            config_dict['Logging']['save-dir'] = \
                os.path.relpath(os.path.join(base_savedir.base, rel_savedir), start=base_savedir.root_dir)
            run_savedir = slurm_utils.write_conf_to_savedir(config_dict, base_savedir, rel_savedir)
            slurm_utils.write_to_bash_script(idx, base_savedir, run_savedir)

        # Run the experiments with slurm
        if parse_and_submit_to_slurm:
            slurm_utils.run_experiments(len(configs_with_path), base_savedir)
            print(f"{len(configs_with_path)} tasks submitted to slurm scheduler.")
        else:
            warnings.warn(f"\n\nRunning {len(configs_with_path)} experiments serially. "
                          "This will likely take hours if not days! "
                          "It is advised you submit with slurm.\n\n")
            for config_dict, rel_savedir in configs_with_path:
                print('-' * 32, f'### Running experiment: {rel_savedir}', '-' * 32, sep='\n')
                run_experiment(os.path.join(base_savedir.base, rel_savedir, 'config.yaml'), plot=plot_after_run)

    else:
        if dry_run:
            print(f'\nDoing dry-run: Not running single experiment at {config_file_path}.')
            return
        run_experiment(config_file_path, plot=plot_after_run)


if __name__ == "__main__":
    # Typical run: python3 main.py --config ../config/matrix/matrix_default.yaml --matrix --slurm
    parser = argparse.ArgumentParser()
    # Arguments in order from most to least useful:
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file to use')
    parser.add_argument('-m', '--matrix', action='store_true', default=False, help='Parse matrix-config')
    parser.add_argument('-s', '--slurm', action='store_true', default=False, help='Submit batch to slurm')
    parser.add_argument('-d', '--dry_run', action='store_true', default=False, help='Do dry-run with slurm, just parsing configs but not submitting anything or creating files.')
    parser.add_argument('-n', '--no_timestamp', action='store_true', default=False, help="Don't include timestamp in logdir when running with matrix parsing")
    parser.add_argument('-p', '--no_plot', action='store_true', default=False, help='Plot immediately after run finishes')
    parser.add_argument('--no_measurements', action='store_true', default=False, help="Don't run measureents")
    _args = parser.parse_args()

    # assert not (_args.no_slurm_timestamp and not _args.matrix), "Illegal combination of arguments, will never use timestamps when not parsing matrix"
    assert not (_args.slurm and not _args.matrix), "Illegal combination of arguments, cannot submit to slurm without allowing matrix config unpacking." \
                                                   " If the config is not in matrix format, unpacking will not change it..."
    assert not (_args.no_plot and _args.slurm), "Illegal combination of arguments, will plot when using slurm."

    main(config_file_path=_args.config,
         unpack_config_matrix=_args.matrix,
         parse_and_submit_to_slurm=_args.slurm,
         use_timestamp_with_matrix=not _args.no_timestamp,
         dry_run=_args.dry_run,
         plot_after_run=not _args.no_plot,
         no_measurements=_args.no_measurements,
    )
    # run_experiment(config_file_path=_args.config)
