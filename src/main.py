import argparse

from Experiment import Experiment
import slurm_utils
import Logger
import os


def run_experiment(config_file_path: str):
    """Run the experiment corresponding to the config file"""
    # TODO(marius): Add option to clean before training (i.e. removing directory before run)
    # TODO(marius): Add option to throw error if measurements already exist. (Maybe lower level code?)
    # TODO(marius): Add plotting automatically
    print("Loading experiment")
    exp = Experiment(config_file_path)
    print("Training NN")
    exp.train()
    print("Running measurements")
    exp.do_measurements_on_checkpoints()


def main(config_file_path: str, parse_and_submit_to_slurm: bool, use_timestamp_with_slurm: bool, slurm_dry_run: bool):
    if parse_and_submit_to_slurm:
        print(f"Parsing matrix config at {config_file_path} and submitting to slurm.")
        configs_with_path, parent_savedir = slurm_utils.parse_config_matrix(config_file_path)
        if slurm_dry_run:
            print(f'\nDoing dry-run: Not submitting to slurm nor creating savedirs: {len(configs_with_path)} jobs.')
            return
        base_savedir = Logger.SaveDirs(parent_savedir, timestamp_subdir=use_timestamp_with_slurm)

        for idx, (config_dict, rel_savedir) in enumerate(configs_with_path):
            config_dict['Logging']['save-dir'] = \
                os.path.relpath(os.path.join(base_savedir.base, rel_savedir), start=base_savedir.root_dir)
            run_savedir = slurm_utils.write_conf_to_savedir(config_dict, base_savedir, rel_savedir)
            slurm_utils.write_to_bash_script(idx, base_savedir, run_savedir)
        slurm_utils.run_experiments(len(configs_with_path), base_savedir)
        print(f"{len(configs_with_path)} tasks submitted to slurm scheduler.")

    else:
        run_experiment(config_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file to use')
    parser.add_argument('-s', '--slurm', action='store_true', default=False, help='Parse matrix-config and submit batch to slurm')
    parser.add_argument('-n', '--no_slurm_timestamp', action='store_true', default=False, help="Don't include timestamp in logdir when running with slurm")
    parser.add_argument('-d', '--slurm_dry_run', action='store_true', default=False, help='Do dry-run with slurm, just parsing configs but not submitting anything or creating files.')
    _args = parser.parse_args()

    main(config_file_path=_args.config,
         parse_and_submit_to_slurm=_args.slurm,
         use_timestamp_with_slurm=(_args.slurm and not _args.no_slurm_timestamp),
         slurm_dry_run=(_args.slurm and _args.slurm_dry_run),
         )
    # run_experiment(config_file_path=_args.config)
