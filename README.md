# Feature Learning in deep classifiers through Intermediate Neural Collapse

This repository contains the code for the experiments in 
> [**Feature Learning in deep classifiers through Intermediate Neural Collapse**,
Rangamani*, Lindegaard*, Galanti, Poggio](https://openreview.net/forum?id=XbggSPNB9W)

published at [ICML 2023](https://icml.cc/Conferences/2023/).

This branch ([ICML2023](https://github.com/mariuslindegaard/Intermediate_Neural_Collapse/tree/ICML2023)) contains
the code relevant for the paper at time of conference publication.
The master branch ([master](https://github.com/mariuslindegaard/Intermediate_Neural_Collapse)) may contain more updated
code, more features (e.g. more datasets) and configs, but might as a consequence be more difficult to navigate.

## Contents
 - Running the experiments
 - Brief overview of the code structure
 - Some details on config files and run options

## Running the experiments

The python dependencies of the project are listed in `environment.yaml`.

Working with this repository is easiest with [conda](https://docs.conda.io/en/latest/miniconda.html) installed.
Additionally, in order to be able to run these experiments in parallel,
the program should be run on a machine (cluster) with [Slurm workload scheduler](https://slurm.schedmd.com/)
(*with GPUs available*, otherwise the job requirements are never satisfied).

Assuming conda is installed, the following commands will run a simple experiment run:
```shell
# Create and activate the conda environment
conda env create -f environment.yaml
conda activate nc

# Run a simple test
cd src
python3 main.py --config ../config/debug.yaml
```
which will give checkpoints, measurements, and plots in `logs/debug/`.

The actual experiments are time-consuming enough that it is *highly* advisable to use slurm on a gpu cluster.
To run the actual experiments in slurm, run
```shell
python3 main.py --config ../config/matrix/matrix_papyan_mseloss.yaml  --matrix --slurm  # For resnets
python3 main.py --config ../config/matrix/matrix_customnets_param_search.yaml --matrix --slurm  # For other networks
```
which will send the jobs to the slurm scheduler.

To conserve computational resources, it can be useful to first run a minimal example:
```shell
python3 main.py --config ../config/matrix_debug.yaml --matrix --slurm
```

The experiments can also run without the slurm scheduler by dropping the `--slurm` flag, but will then run serially.

## File structure and code overview

Any experiment is run by running and specifying a config file: `main.py --config path/to/config.yaml`.

In its simplest form this will:
 1. Create an `Experiment` object (from `Experiment.py`), passing the config file from the given path.
 2. After the config file has been parsed, the `Experiment` contains all
specifications of the experiment in its attributes:
    1. The dataset (a `DatasetWrapper`)
    2. The model (a `Models.ForwardHookedOutput`)
    3. The optimizer (a `WrappedOptimizer`)
    4. A logging-handler (a `Logger`)
    5. The IDs of the measures (a dictionairy pointing to the different `Measure` classes)
 3. Then, handled by the Experiment functions:
    1. The model is trained using the specified dataset (calling `experiment.train()`), saving checkpoints along the way.
    2. For each checkpoint, for each specified layer, the `Measure.measure(...)` methods are called, producing a `measure_name.csv` file.
 4. Finally, `NCPlotter.plot_runs` runs where the measures are saved, creating the relevant plots from the measurement data.

All the files will be put in a log-file specified by the `Logging: log-dir: logs/dir/path`, typically under `logs/`.

### Inspecting the experiment in debug-mode

As a starting point, `src/simple_debugging_tool.py` is provided to show a simple way to inspect the models and embeddings.
Check the file for more details on how to load already trained models. It should mesh well with the
(TODO) accompanying zip-file with the trained models on which measurements for the paper were performed.

### Parameter search in a single config file

Some of the config files (typically `config/matrix/somename.yaml`) contain a `Matrix` parameter.
When calling `main.py` with the `--matrix` (or `-m`) flag, this will generate subfolders named with the
hyperparameters/config used in this specific run, and place the relevant config file in the correct subfolder.

### Flags 
Depending on the additional flags set, either:
 - If no additional flags, the runs will be performed sequentially. *This is not reccommended as it might take days to run.*
 - If the `-s` (`--slurm`) flag is set, the tasks will be submitted in parallel to a local slurm job manager.
   *This is the recommended way to run the experiments.*
 - A dry run will be performed if `-d` (`--dry_run`) is set, showing details of the different runs and configs
   without actually running the experiments nor creating the folder structure.
 - For other flags, refer to `src/main.py` or run `python3 main.py -h`.
