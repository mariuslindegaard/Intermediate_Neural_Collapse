import argparse

from Experiment import Experiment

def run_experiment(config_file_path: str):
    """Run the experiment corresponding to the config file"""
    # TODO(marius): Add option to clean before training (i.e. removing directory before run)
    # TODO(marius): Add option to throw error if measurements already exist. (Maybe lower level code?)
    # TODO(marius): Add plotting automatically
    # TODO(marius): Add support for submitting a slurm job
    print("Loading experiment")
    exp = Experiment(config_file_path)
    print("Training NN")
    exp.train()
    print("Running measurements")
    exp.do_measurements_on_checkpoints()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file to use')
    _args = parser.parse_args()
    run_experiment(config_file_path=_args.config)
