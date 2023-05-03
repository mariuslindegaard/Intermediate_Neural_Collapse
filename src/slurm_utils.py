import itertools
import os
import copy
import warnings
import subprocess
from typing import Iterator, Tuple, List, Dict, Any

import yaml

import Logger


def parse_config_matrix(config_file_path: str) -> Tuple[List[Tuple[Dict, str]], str]:
    """Parse a config file formatted as a matrix

    :param config_file_path: Path to matrix config file
    :returns: Tuple with (a list of (tuples of yaml dictionaries and its relative savedir to savepath)),
                          and the savepath where they're all stored.
    """
    with open(config_file_path, 'r') as config_file:
        config_matrix: Dict = yaml.safe_load(config_file)

    base_savepath = config_matrix['Logging']['save-dir']

    matrix_params = config_matrix.get('Matrix', None)
    if matrix_params is None:
        return [(config_matrix, "")], base_savepath

    base_config = copy.deepcopy(config_matrix)
    base_config.pop('Matrix')

    config_list = []
    for config_edits, rel_savepath in _matrix_config_parser(config_matrix['Matrix']):
        print(rel_savepath, config_edits, sep='\n', end='\n---\n')
        new_config = copy.deepcopy(base_config)
        _apply_configdict_edits(new_config, config_edits)
        new_config['Logging']['save-dir'] = rel_savepath
        config_list.append((new_config, rel_savepath))

    return config_list, base_savepath


def _apply_configdict_edits(config_dict: Dict, config_edits: Dict[str, Any],
                            _base: Tuple[str, ...] = tuple()) -> None:
    """Apply edits inplace to input config dict

    :param config_dict: Config dict to apply edits to
    :param config_edits: Edits to apply
    :return: None
    """
    for key, val in config_edits.items():
        if type(val) is dict:
            if key not in config_dict:
                config_dict[key] = {}
            _apply_configdict_edits(config_dict[key], val, _base + (key,))
        else:
            config_dict[key] = val


def _merge_config_dicts(*config_dicts: Dict) -> Dict:
    """Merge config dicts, with later dicts overwriting earlier ones. Does not modify input dicts.

    :param config_dicts: Config dicts to merge, in order of precedence (i.e. later dicts overwrite earlier ones)
    :return: Merged config dict
    """
    if len(config_dicts) == 0:
        return {}
    elif len(config_dicts) == 1:
        return config_dicts[0].copy()
    else:
        new_dict = _merge_config_dicts(*config_dicts[:-1])
        _apply_configdict_edits(new_dict, config_dicts[-1])
        return new_dict


def _path_amendment(x: Dict[str, Any], key: str) -> str:
    """Amendment to path for specific point in config matrix, specifying where to save

    :param x: Config dict
    :param key: Key to amend path with
    :return: Amendment to path
    """
    # if key.lower() == 'measures':  # If the only difference is in the measures, let them run since they output to different files
    #     return "m"
    if key.lower() in ('dataset-id', 'model-name'):
        return str(x[key])
    elif key.lower() == 'weight-decay':
        return f'wd_{x[key]}'
    else:
        return f'{key}_{x[key]}'


def _matrix_config_parser(config_matrix: Dict) -> Iterator[Tuple[Dict[str, Any], str]]:
    """Get iterator over all configs as specified by matrix config

    :param config_matrix: Config file dict (at root of file) with 'Matrix' parameter
    :return: Iterator over all configs in dict form with generated string of savepath
    """
    # pools contains a list of list of tuples (config_edits, outer_key_for_this_edit). Outer product gives matrix.
    pools: List[List[Tuple[Dict[str, Any], str]]] = []

    # Handle recursing through tree
    for key, val in config_matrix.items():
        if key == '_Exclusive':
            continue
        if type(val) is dict:
            # for edits, path_add in _matrix_config_parser(val):
            pools.append(
                list(map(lambda x: ({key: x[0]}, x[1]), _matrix_config_parser(val)))
            )
        elif type(val) is list:
            # Parse values to run matrix over
            pool_add_tmp: List[Dict[str, Any]] = []
            for elem in val:
                pool_add_tmp.append({key: elem})

            if len(pool_add_tmp) > 1:  # Add relevant path amendment
                pools.append(list(map(lambda x: (x, _path_amendment(x, key)), pool_add_tmp)))
            else:
                pools.append(list(map(lambda x: (x, ""), pool_add_tmp)))

        else:
            raise ValueError(
                f"Expected all leaves of matrix config to be list, but {key}: {val} is type {type(val)}.")

    # Handle exclusive configs
    exclusive_pool = []
    if '_Exclusive' in config_matrix.keys():
        for conf_id, conf in config_matrix['_Exclusive'].items():
            exclusive_pool.extend(
                list(map(lambda x: (x[0], f'{os.path.join(conf_id, x[1])}'),
                         list(_matrix_config_parser(conf))))
            )
    # Take the outer product of all edits and yield
    for edits in itertools.product(*pools):
        edits: Tuple[Tuple[Dict[str, Any], str]]
        current_edits: Dict[str, Any] = {}
        current_path = ''
        for edit, path_amendment in edits:
            current_edits = _merge_config_dicts(current_edits, edit)
            current_path = os.path.join(current_path, path_amendment)

        for exclusive_edits, exclusive_path in exclusive_pool:
            yield _merge_config_dicts(current_edits, exclusive_edits), os.path.join(exclusive_path, current_path)

        if not exclusive_pool:
            yield current_edits, current_path


def write_conf_to_savedir(config_dict: Dict, parent_savedir: Logger.SaveDirs, rel_savedir: str) -> Logger.SaveDirs:
    """Write config dict to relevant savedir.

    :param config_dict: Config dict to write
    :param parent_savedir: Root savedir
    :param rel_savedir: Relative savedir from root savedir
    :return: SaveDirs object for this run's savedir.
    """
    new_savedir = copy.deepcopy(parent_savedir)
    new_savedir.force_new_base_path(os.path.join(new_savedir.base, rel_savedir))

    if os.path.exists(new_savedir.config):
        # config_is_equal = filecmp.cmp(config_dict, target_path)
        # raise FileExistsError(f"Config file at {target_path} exists!")
        warnings.warn(f"Config file at {new_savedir.config} exists! Overwriting and continuing execution.....")

    os.makedirs(new_savedir.base, exist_ok=True)
    with open(new_savedir.config, 'w+') as target_file:
        yaml.safe_dump(config_dict, target_file)

    return new_savedir


def write_to_bash_script(idx: int, base_savedir: Logger.SaveDirs, run_savedir: Logger.SaveDirs) -> str:
    """Write to bash script to be run by slurm.

    :param idx: Job idx
    :param base_savedir: Base savedir
    :param run_savedir: Savedir for this run
    :return: Path to script
    """
    script_path = os.path.join(base_savedir.base, '__slurm/jobs', f'{idx}.sh')
    script_content = _JOB_SCRIPT_STUMP.format(
        os.path.join(base_savedir.root_dir, 'src/main.py'),
        run_savedir.config
    )

    os.makedirs(os.path.join(base_savedir.base, '__slurm/jobs'), exist_ok=True)
    with open(script_path, 'w+') as script_f:
        script_f.write(script_content)

    # Write path to csv-file in __slurm
    with open(os.path.join(base_savedir.base, '__slurm/path_job_map.csv'), 'a+') as csv_f:
        csv_f.write(f'{idx},{os.path.relpath(run_savedir.base, base_savedir.base)}\n')

    return script_path


def run_experiments(num_scripts: int, base_savedir: Logger.SaveDirs):
    """Create sbatch and submit to slurm!

    :param num_scripts:
    :param base_savedir:
    :return:
    """
    # Create sbatch file
    sbatch_script_path = os.path.join(base_savedir.base, '__slurm/execute_array.sh')
    sbatch_script_content = _SBATCH_SCRIPT_STUMP

    os.makedirs(os.path.join(base_savedir.base, '__slurm'), exist_ok=True)
    with open(sbatch_script_path, 'w+') as script_f:
        script_f.write(sbatch_script_content)
    os.chmod(sbatch_script_path, 777)

    # Execute sbatch file
    sbatch_script_exec_command = _SBATCH_SCRIPT_EXECUTE_COMMAND.format(
        start_idx=0, end_idx=num_scripts-1,
        configs_path_dir=os.path.join(base_savedir.base, '__slurm/jobs'),
        sbatch_file_path=sbatch_script_path
    )
    # Create symlink for datasets datasets
    os.symlink(os.path.join(base_savedir.root_dir, 'datasets'), os.path.join(base_savedir.base, '__slurm/jobs/datasets'))
    print("Executing", sbatch_script_exec_command)
    try:
        process = subprocess.run(
            sbatch_script_exec_command,
            shell=True, check=True,
            cwd=os.path.join(base_savedir.base, '__slurm'),
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print("Sbatch submission failed with the following output to stdout and stderr:")
        print(e.stdout)
        print(e.stderr)
        raise e
    # 'sbatch --array={start_idx}-{end_idx} --export=configs_path_dir={configs_path_dir} {sbatch_file_path}'


_JOB_SCRIPT_STUMP = \
    """#!/bin/bash

source ~/.conda_init
conda activate nc

echo
echo "~~~ RUNNING EXPERIMENT! ~~~"
echo
echo "Running with config file {1:}"
python3 {0:} --config {1:}
echo
echo
echo "~~~ FINISHED EXPERIMENT! ~~~"
"""

_SBATCH_SCRIPT_STUMP = \
    '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1  # --constraint=40GB
#SBATCH --constraint=high-capacity
#SBATCH --time=47:59:00
#SBATCH --exclude=node021,node086
#SBATCH --mem=24000
#SBATCH --requeue
# #SBATCH --qos=cbmm  # TODO(marius): Remove cbmm-identifier for submission
# #SBATCH -p cbmm
# #SBATCH --output=./output.log

# #SBATCH --mail-user=lindegrd@mit.edu  # TODO(marius): Remove email identifier
# #SBATCH --mail-type=ALL

# memory 1000 MiB
# gpu_mem 10904 MiB

# Execute with:
#     sbatch --array=start_idx-stop_idx --export=configs_path_dir=dir_containing_idx_dot_sh this_file.sh
#   For example, with the standard config and a total of 12 jobs when in the __slurm directory:
#     sbatch --array=0-11 --export=configs_path_dir=jobs execute_array.sh

date;hostname;id;pwd
echo
echo "running script ${SLURM_ARRAY_TASK_ID}.sh in ${configs_path_dir}"

cd "${configs_path_dir}"
chmod +x "${SLURM_ARRAY_TASK_ID}.sh"
srun -n 1 "${SLURM_ARRAY_TASK_ID}.sh"
'''

_SBATCH_SCRIPT_EXECUTE_COMMAND = \
    'sbatch --array={start_idx}-{end_idx} --export=configs_path_dir={configs_path_dir} {sbatch_file_path}'
