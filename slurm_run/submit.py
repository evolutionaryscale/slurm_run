"""Utilities for programmatic training job submissions to SLURM clusters."""

import json
import os
import pickle
import re
import shlex
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from getpass import getuser
from pathlib import Path
from typing import Union

from .utils import temporarily_unset_slurm_env_vars, test_if_pixi_lock_file_is_valid

BASE_CONFIG_FILE_TEMPLATE = """
import pickle
import os
from dataclasses import asdict, dataclass

def config():
    if os.environ.get("SLURM_ARRAY_TASK_ID", ""):
        config_suffix = "_" + os.environ.get('SLURM_ARRAY_TASK_ID', '')
    else:
        config_suffix = "_1"
    config_path = "{config_prefix}" + config_suffix + ".cfg.pkl"
    with open(config_path, "rb") as f:
        loaded = pickle.load(f)
"""

CONFIG_FILE_TEMPLATE = (
    BASE_CONFIG_FILE_TEMPLATE
    + """
        if isinstance(loaded, dict):
            return loaded
        else:
            return asdict(loaded)

"""
)

CONFIG_FILE_TEMPLATE_WITHOUT_DICT_CAST = (
    BASE_CONFIG_FILE_TEMPLATE
    + """
        return loaded
"""
)


SBATCH = """#!/usr/bin/env bash
{prelude}

if [[ -n $SLURM_STEP_ID ]] || [[ {no_srun} -eq 1 ]]; then
    export WORKDIR=$(mktemp -d)
    cd $WORKDIR
    tar xf {snapshot} --force-local
    export WANDB_MODE={wandb}
    ulimit -n $(ulimit -Hn)  # lift open-file limit.
    export EVORUN_SBATCH={sbatch_path}
    export EVORUN_SNAPSHOT={snapshot}
    # Setup pixi environment
    # NOTE(rverkuil): PIXI_ENVIRONMENT_NAME *will* pass through, controlling the environment that is used.
    unset PIXI_ENVIRONMENT_PLATFORMS PIXI_PROJECT_MANIFEST PIXI_PROJECT_NAME PIXI_PROJECT_ROOT PIXI_PROJECT_VERSION PIXI_PROMPT
    # Log some stuff to ease debugging
    env
    hostname
    echo Building pixi environment...
    output=$(
        timeout 10m pixi shell-hook --locked
    )
    exit_code=$?
    if [[ $exit_code -eq 124 ]]; then
        echo "pixi shell-hook timed out, likely due to hanging pixi install"
        exit 124
    fi
    if [[ $exit_code -ne 0 ]]; then
        echo "pixi install failed with exit code $exit_code"
        exit $exit_code
    fi
    source <(echo "$output")
    # Clean up pixi environment, remove the env as well if it's in /tmp
    echo Running script at {sbatch_path}, with code image {snapshot} in $(pwd)...
    {command} & PID=$!
    trap "kill -TERM $PID; wait $PID" TERM
    trap "kill -USR2 $PID; wait $PID" USR2
    wait $PID
    exit_code=$?
    # notify user if applicable, needs pixi env to still exist, so cant use trap, meaning scancel would skip this
    if [[ {notify} -eq 1 ]]; then
        # no_srun=1: SLURM_PROCID is usually unset -> this branch runs once
        # no_srun=0: only task with SLURM_PROCID=0 runs this
        # NOTE: might send notification early if other proc still running
        if [[ -z "$SLURM_PROCID" || "$SLURM_PROCID" -eq 0 ]]; then
            python $WORKDIR/scripts/slack/job_notifier.py --job-id "$SLURM_JOB_ID" --exit-code "$exit_code"
        fi
    fi
    echo 'Cleaning up pixi'
    rm -rf $WORKDIR

else
    # `--kill-on-bad-exit` kills all tasks if any one task fails.
    # `--label` Prepend task number to lines of stdout/err.  To filter an output file for a task, run:
    #     `grep -P "^$TASK_NUM_WITH_LEADING_SPACES: " path/to/log.out`
    srun --kill-on-bad-exit --label {sbatch_path} & wait  # We need this so traps and restarting works...
fi
"""

SBATCH_NO_ENV_SETUP = """#!/usr/bin/env bash
{prelude}

if [[ -n $SLURM_STEP_ID ]] || [[ {no_srun} -eq 1 ]]; then
    export WORKDIR=$(mktemp -d)
    cd $WORKDIR
    tar xf {snapshot} --force-local
    export WANDB_MODE={wandb}
    ulimit -n $(ulimit -Hn)  # lift open-file limit.
    export EVORUN_SBATCH={sbatch_path}
    export EVORUN_SNAPSHOT={snapshot}
    # Log some stuff to ease debugging
    env
    hostname

    echo Running script at {sbatch_path}, with code image {snapshot} in $(pwd)...
    {command} & PID=$!
    trap "kill -TERM $PID; wait $PID" TERM
    trap "kill -USR2 $PID; wait $PID" USR2
    wait $PID
    exit_code=$?
    echo 'Cleaning up'
    rm -rf $WORKDIR

else
    # `--kill-on-bad-exit` kills all tasks if any one task fails.
    # `--label` Prepend task number to lines of stdout/err.  To filter an output file for a task, run:
    #     `grep -P "^$TASK_NUM_WITH_LEADING_SPACES: " path/to/log.out`
    srun --kill-on-bad-exit --label {sbatch_path} & wait  # We need this so traps and restarting works...
fi
"""

SBATCH_RAY = """#!/usr/bin/env bash
{prelude}

export WORKDIR=$(mktemp -d -p {shared_env_dir})
cd $WORKDIR
tar xf {snapshot} --force-local
export WANDB_MODE={wandb}
ulimit -n $(ulimit -Hn)  # lift open-file limit.
export EVORUN_SBATCH={sbatch_path}
export EVORUN_SNAPSHOT={snapshot}
# Setup pixi environment
# NOTE(rverkuil): PIXI_ENVIRONMENT_NAME *will* pass through, controlling the environment that is used.
unset PIXI_ENVIRONMENT_PLATFORMS PIXI_PROJECT_MANIFEST PIXI_PROJECT_NAME PIXI_PROJECT_ROOT PIXI_PROJECT_VERSION PIXI_PROMPT
# Log some stuff to ease debugging
env
echo Building pixi environment...
source <(./bin/pixi shell-hook --locked)
# Clean up pixi environment, remove the env as well if it's in /tmp
trap "nohup rm -rf $WORKDIR >/dev/null 2>&1 & disown" EXIT INT TERM QUIT USR2
echo Running script at {sbatch_path}, with code image {snapshot} in $(pwd)...

evolutionaryscale/utils/sbatch_ray.sh {command}
rm -rf $WORKDIR
"""


MAIN_FUNCTIONS = {
    "orz": "projects/archimedes/reasoning/orz/experiments/ppo/train.py",
    "archimedes": "projects/archimedes/joint_training/train.py",
    "train": "projects/esm3/pretraining/train.py",
    "dpo": "projects/esm3/dpo/dpo.py",
}


def _format_slurm_comment(job_type: str, proj: str) -> str:
    # Reserved for formatting the comment string.
    forbidden_chars = ['"', ":"]  # For safe JSON serialization.
    for s in [job_type, proj]:
        for c in forbidden_chars:
            assert c not in s, f"Comment string '{s}' should not contain {c}"
    return ":".join([job_type, proj])


def _is_training_job(command: str) -> bool:
    # Detect this is a single training job, a bit hacky but better than alternatives
    # TODO(jenna): consider better method for detecting training jobs
    return "train.py" in command or ".train" in command


def _guess_proj_from_command(command: str) -> str:
    # Try to guess the project of a slurm command based on the path
    # of the program or data.
    m = re.search(r"projects/([^/]+)", command)
    if m:
        return m.group(1)
    m = re.search(r"projects\.([^\.]+)", command)
    if m:
        return m.group(1)
    # Default.
    return "other"


@dataclass
class SubmissionConfig:
    run_name: str
    partition: str
    gpus: int
    nice: int = 0
    time_limit: str = "7-0"
    retry: bool = False
    dry_run: bool = False
    no_wandb: bool = False
    ray: bool = False
    array_limit: int | None = None
    wait_until_completion: bool = False


def _slurm_job_id_string():
    return '$([ -z "$SLURM_ARRAY_TASK_ID" ] && echo $SLURM_JOB_ID || echo ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}) '


def _slurm_retry_trap():
    return (
        "SLMT_TRAP_FN() {\n"
        "echo Trapped: restarting\n"
        f"scontrol requeue {_slurm_job_id_string()}\n"
        "}\n"
        "trap SLMT_TRAP_FN USR2\n"
    )


def _maybe_path(lst) -> Path:
    for p in lst:
        if (p := Path(p)).exists():
            return p
    else:
        raise RuntimeError(f"No path found in {lst}")


def _get_user_jobs(starttime: str = "now-4weeks", user: str | None = None) -> dict:
    """Get all jobinfo for the current user. Always lists in order of job id."""
    output = subprocess.check_output(
        ["sacct", "-u", user or getuser(), "--json", "--starttime", starttime],
        text=True,
    )
    job_info = json.loads(output)["jobs"]

    return job_info


def list_jobs_for_user(
    starttime: str,
    user: str | None = None,
    columns: str = "JobID,JobName,Partition,State,Elapsed",
) -> list[dict]:
    """Dict of selected job info for the current user"""
    user_jobs = _get_user_jobs(starttime, user)
    job_data = []

    for job in user_jobs:
        wandb_run_name = get_wandb_run_name_from_script(job["submit_line"])

        job_info = {
            "JobID": job["job_id"],
            "ArrayJobID": job["array"]["job_id"],
            "JobName": job["name"],
            "Tag": job["comment"]["job"].strip('"'),
            "Partition": job["partition"],
            "State": ",".join(
                job["state"]["current"]
            ),  # Note: unclear why this is a list
            "Command": job["submit_line"],
            "Stdout": job["stdout_expanded"],
            "Stderr": job["stderr_expanded"],
            "Elapsed": str(timedelta(seconds=int(job["time"]["elapsed"]))),
            "WandBJobName": wandb_run_name,
        }
        job_data.append(job_info)

    if not job_data:
        print(f"No jobs found since starttime {starttime}")
        return []

    # Always filter columns based on user preference (including default)
    requested_columns = [col.strip() for col in columns.split(",")]
    available_columns = job_data[0].keys() if job_data else []

    # Validate requested columns
    invalid_columns = [col for col in requested_columns if col not in available_columns]
    if invalid_columns:
        raise ValueError(
            f"Invalid columns: {invalid_columns}. Available columns: {list(available_columns)}"
        )

    # Filter columns for each job
    filtered_job_data = []
    for job in job_data:
        filtered_job = {col: job[col] for col in requested_columns}
        filtered_job_data.append(filtered_job)
    job_data = filtered_job_data

    return job_data


def _rerun_job(job_dict: dict) -> str:
    """Rerun a job given its job info"""
    if _job_running(job_dict):
        raise RuntimeError(
            f"Job {job_dict['job_id']} is already {job_dict['state']['current'][0]}, not rerunning. Run `scancel {job_dict['job_id']}` to cancel this job if desired."
        )
    if job_dict["name"]:  # If job is named, check against active jobs
        active_job = job_name_active(job_dict["name"])
        if active_job:
            raise RuntimeError(
                f"A job with name {active_job['name']} (id={active_job['job_id']}) is already {active_job['state']['current'][0]}, please select another name."
            )
    print(f"Rerunning job {job_dict['job_id']} cmd: {job_dict['submit_line']}")

    output = subprocess.check_output(job_dict["submit_line"], shell=True).decode(
        "ascii"
    )
    job_id = output.strip().split()[-1]

    return job_id


def _job_running(job_dict: dict) -> bool:
    """Check job dict for running or pending state"""
    return job_dict["state"]["current"][0] in ["RUNNING", "PENDING"]


def get_wandb_run_name_from_script(submit_line: str) -> str:
    """
    Extract WandB run name from a job's submit line by finding and reading the script file.

    Parameters
    ----------
    submit_line : str
        The submit line from the job (contains sbatch command)

    Returns
    -------
    str
        WandB run name or appropriate error message. Returns:
        - The actual run name if found
        - "N/A" if no submit line provided or logger.run_name not found in script
        - "Error: No script found" if no script path in submit line
        - "Error: Script file not found" if script file doesn't exist
        - "Error: Reading script file" if file reading fails
    """
    if not submit_line:
        return "N/A"

    # Look for .sh script path in submit_line (sbatch <script>)
    match = re.search(r"sbatch\s+([^\s]+\.sh)", submit_line)
    if not match:
        return "Error: No script found"

    script_path = match.group(1)

    if not os.path.exists(script_path):
        return "Error: Script file not found"

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for logger.run_name=<name> pattern
        match = re.search(r"logger\.run_name=([^\s\n]+)", content)
        if match:
            return match.group(1)
        else:
            return "N/A"

    except Exception:
        return "Error: Reading script file"


def job_name_active(job_name: str) -> Union[dict, None]:
    """Check if a job name is already running or pending"""
    user_jobs = _get_user_jobs()
    for job in reversed(user_jobs):
        if job["name"] == job_name and _job_running(job):
            return job
    return None


def current_job_partition() -> str:
    """Get the partition of the current job or default to midpri if not set"""
    if "SLURM_JOB_PARTITION" not in os.environ:
        return "midpri"
    return os.environ["SLURM_JOB_PARTITION"]


def rerun_job_by_id(job_id: int) -> list[str]:
    """Rerun a job by its job id. Will work for both job ids and array ids."""
    user_jobs = _get_user_jobs()

    rerun_ids = []
    for job in user_jobs:
        if job["job_id"] == job_id or job["array"]["job_id"] == job_id:
            rerun_ids.append(_rerun_job(job))

    if not rerun_ids:
        # If we are here, we can not find the job to rerun.
        raise RuntimeError(f"No jobs with id or array id {job_id} found.")

    return rerun_ids


def rerun_jobs_by_tag(tag: str) -> list[str]:
    """Rerun jobs tied to a specific tag. Will attempt to run all jobs found for the tag and stop at first failure."""
    user_jobs = _get_user_jobs()

    rerun_ids = []
    for job in user_jobs:
        if job["comment"]["job"] == tag:
            print(job)
            rerun_ids.append(_rerun_job(job))

    if not rerun_ids:
        # If we are here, we can not find the job to rerun.
        raise RuntimeError(
            f"No jobs with tag {tag} found. If job is still running, tag will not be set until it finishes."
        )

    return rerun_ids


def rerun_jobs_by_name(name: str) -> list[str]:
    """Rerun jobs tied to a specific name. If there are multiple in the history, the last one(s) will be rerun."""
    user_jobs = _get_user_jobs()
    for job in reversed(user_jobs):
        if job["name"] == name:
            if job["array"]["job_id"] != 0:
                # This job is part of an array, so also rerun the other jobs in the array
                return rerun_job_by_id(job["array"]["job_id"])
            else:
                return [_rerun_job(job)]
    # If we are here, we can not find the job to rerun.
    raise RuntimeError(f"No jobs with name {name} found.")


def slurm_run(
    command: str,
    image: Path,
    sbatch_dest: Path,
    run_name: str | None = None,
    partition: str | None = None,
    gpus: int = 8,
    cpus: int = 8,
    ntasks: int | None = None,
    nice: int = 0,
    time_limit: str = "7-0",
    dry_run: bool = False,
    array: int = 0,
    array_limit: int | None = None,
    retry: bool = False,
    verbose: bool = False,
    no_wandb: bool = False,
    tag: str | None = None,
    ray: bool = False,  # run with ray-on-slurm
    exclusive: bool = False,  # run with exclusive access to the nodes
    no_srun: bool = False,
    dependency: str = "",
    wait_until_completion: bool = False,
    constraint: str = "h100-reserved",
    comment: str | None = None,
    notify: bool = False,
    no_env_setup: bool = False,
) -> str:
    # Warn if PIXI_CACHE_DIR is set (no longer needed on new filesystem)
    if "PIXI_CACHE_DIR" in os.environ:
        print(
            "WARNING: PIXI_CACHE_DIR environment variable is set but no longer needed."
        )
        print(
            "Please remove it from your environment (e.g., unset PIXI_CACHE_DIR or remove from .bashrc)."
        )
        print()

    if not no_env_setup:
        assert test_if_pixi_lock_file_is_valid(), "Pixi lock file is not valid, please run `pixi install` or `pixi shell` to fix this."
    else:
        assert not ray, "Ray jobs cannot be run without pixi environment setup. Please set no_env_setup to True."

    if exclusive and gpus % 8 != 0:
        raise ValueError(
            "Exclusive access should only be used with whole nodes (gpus % 8 == 0), please set gpus to a multiple of 8."
        )

    if gpus <= 0:
        # This is a CPU only job.
        # Make sure ntasks is specified.
        assert ntasks and ntasks > 0, "Must specify ntasks for CPU only jobs."
    elif ntasks is not None:
        # For GPU jobs, ntasks is ignored.
        # Warn user about this.
        print("Warning: when gpus > 0, ntasks parameter is ignored.")

    if partition is None:
        partition = "h100-reserved"

    if run_name:
        if not array:
            if _is_training_job(command):
                assert (
                    "logger.run_name=" not in command
                ), "logger.run_name already set by --run-name flag, please remove this config override"
                command += f" logger.run_name={run_name}"
    else:
        print(
            f"No job name specified, defaulting to sbatch filename '{sbatch_dest.name}'"
        )
        run_name = sbatch_dest.name

    log_dest = sbatch_dest.parent / ("slurm-%A_%a.out" if array > 0 else "slurm-%j.out")
    if ray:
        assert gpus % 8 == 0, "Ray jobs must use entire nodes, e.g. gpus % 8 == 0"
        nodes = gpus // 8
        # Don't allocate as many cpus to Ray jobs for now, but adjust as needed.
        gpu_res = f"-c {cpus*8} --gres gpu:8 -N {nodes}"
        slurm_flags = f"#SBATCH -p {partition} -t {time_limit} -J {run_name} --nice={nice} -o {log_dest} {gpu_res} --signal B:USR2@60 --requeue --open-mode=append"
    else:
        if gpus <= 0:
            # CPU only job. Simply specify how many tasks we need.
            gpu_res = f"-n {ntasks}"
        elif gpus < 8:
            # For smaller training jobs ... NCCL only supports --gres here
            gpu_res = f"-N 1 -n {gpus} --gres gpu:{gpus}"
        elif gpus % 8 == 0:
            # Larger training jobs, allocated as whole nodes
            gpu_res = f"-N {gpus // 8} --ntasks-per-node 8 --gres gpu:8"
        else:
            # Revisit this case because it will require understanding what
            # the user is really trying to do. Any jobs that require NCCL
            # communication won't work without using --gres and its wasteful
            # to allocate a whole node for a partial job. Any jobs that don't
            # need communication will work but its a different usecase.
            raise ValueError(
                "Large jobs should be allocated as whole nodes, e.g. gpus % 8 == 0"
            )

        slurm_flags = f"#SBATCH {gpu_res} --cpus-per-task {cpus} -p {partition} -t {time_limit} -J {run_name} --nice={nice} -o {log_dest} --signal B:USR2@60 --requeue --open-mode=append"

    if array > 0:
        assert not ray, "Ray jobs cannot be array jobs"
        slurm_flags += f" --array=1-{array}"
        if array_limit is not None:
            slurm_flags += f"%{array_limit}"

    if exclusive:
        slurm_flags += " --exclusive --mem=0"
    else:
        # By setting this flag, we are requesting 16 GB * 8 * 8 = 1024 GB of memory per node.
        slurm_flags += " --mem-per-cpu=16G"

    if dependency:
        # Job has dependency.
        slurm_flags += f" --dependency {dependency}"

    if wait_until_completion:
        slurm_flags += " --wait"

    slurm_flags += f" --constraint {constraint}"

    if comment:
        assert (
            '"' not in comment
        ), 'Please make sure comment parameter does NOT include " chars.'
        slurm_flags += f' --comment "{comment}"'

    if no_env_setup:
        template = SBATCH_NO_ENV_SETUP
    elif ray:
        template = SBATCH_RAY
    else:
        template = SBATCH
    sbatch = template.format(
        prelude=f"{slurm_flags}\n{_slurm_retry_trap() if retry else ''}",
        snapshot=shlex.quote(str(image)),
        sbatch_path=shlex.quote(str(sbatch_dest)),
        command=str(command),
        wandb="disabled" if no_wandb else "online",
        shared_env_dir=sbatch_dest.parent,
        no_srun="1" if no_srun else "0",
        notify="1" if notify else "0",
    )
    sbatch_dest.write_text(sbatch)
    sbatch_dest.chmod(sbatch_dest.stat().st_mode | stat.S_IXUSR)

    if dry_run:
        if verbose:
            print(sbatch_dest)
            print(
                f"sbatch script saved to {sbatch_dest}, check it out!", file=sys.stderr
            )
        return ""  # Try run does not have job_id.

    if verbose:
        print(f"Running sbatch at {sbatch_dest}", file=sys.stderr)

    # NOTE(rverkuil): This context manager temporarily unsets
    # all SLURM environment variables, while we are actively sbatch-ing
    # a new job. This allows users to submit new jobs from within
    # existing jobs. For example, consider the not-infrequent case of a user
    # launching jobs from within an interactive session on a GPU node.
    # Without this context manager, foot-gun behavior occurs - the new
    # job inherits SLURM env vars from the current session that are not
    # explicitly set in the new job. This can lead to unexpected behavior
    # like errors or mysterious job hangs due to insufficient tasks / resources.
    with temporarily_unset_slurm_env_vars():
        # Comment is used as a user-defined tag for a job
        comment_arg = ["--comment", f'"{tag}"'] if tag is not None else []
        output = subprocess.check_output(
            ["sbatch"] + comment_arg + [str(sbatch_dest)], text=True
        )
    job_id = output.strip().split()[-1]
    if array > 0:
        log_dest = str(log_dest).replace("%A", job_id)
    else:
        log_dest = str(log_dest).replace("%j", job_id)
    if verbose:
        print("Submitted batch job", job_id)
        if array > 0:
            for i in range(1, array + 1):
                log_dest_i = log_dest.replace("%a", str(i))
                print(f"Logs for job {i} will be saved to {log_dest_i}")
        else:
            print(f"Logs will be saved to {log_dest}")

    return job_id


def _verify_submission_cfg(submission_cfg: SubmissionConfig):
    assert submission_cfg.partition in [
        "lowpri",
        "midpri",
        "highpri",
        "hero",
        "antihero",
    ]
    assert submission_cfg.gpus > 0
    assert submission_cfg.gpus % 8 == 0 or submission_cfg.gpus < 8


def mkimg(
    e: str,
    ignore: str = "wandb/,*.egg-info/,.pixi/,.git/,slurm/",
    dest: str = ".code_snapshots",
    force: list[str] | None = None,
    require_pixi: bool = True,
):
    """Archives your current directory using fd-find.

    :param e: Comma-separated file types (extensions) to archive.
    :param ignore: Comma-separated directory patterns to ignore.
    :param dest: Where to save outputs to.
    :param force: List of files to force include.
    """
    extensions = [f"-e {ext.strip()}" for ext in e.split(",") if ext.strip()]
    excludes = [
        f"-E '{pattern.strip()}'" for pattern in ignore.split(",") if pattern.strip()
    ]

    fd_cmd_parts = ["fd", "-u"] + extensions + ["--type", "f"] + excludes + ["."]
    fd_cmd = " ".join(fd_cmd_parts)
    fd_links = f"fd -u -t l {' '.join(excludes)}"  # for softlinks of any ext

    # Optionally package pixi to make this hermetic\
    if require_pixi:
        pixi_exe = os.environ["PIXI_EXE"]
        force = force.copy() if force else []
        force.append(pixi_exe)
        transforms = [f"'s|{pixi_exe}|bin/pixi|'"]
    else:
        transforms = []

    forced_files = ("echo " + "\\\\n".join(force)) if force else ""

    now = datetime.now()
    dir_format = now.strftime("%Y%m%d")
    dest_format = now.strftime("%Y%m%dT%H%M%SM%f")

    output_dir = Path(dest) / dir_format
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = str(output_dir / f"{dest_format}.tar.gz")

    # package abspaths... least worst option to squelch warnings: tar: Removing leading `/' ...
    # https://unix.stackexchange.com/questions/59243/tar-removing-leading-from-member-names
    cmd = f"({forced_files}; {fd_cmd}; {fd_links}) |  tar -cP --transform={' --transform='.join(transforms)} -T - -f - | gzip -c > {shlex.quote(tar_path)}"

    subprocess.check_call(cmd, shell=True)

    return tar_path


def dump_code_image(require_pixi: bool = True) -> Path:
    path = _maybe_path([f"/mnt/main0/home/{getuser()}"]) / "slurm"
    path.mkdir(exist_ok=True)
    if require_pixi and not Path("./pixi.toml").exists():
        raise RuntimeError(
            "No pixi.toml found, evorun only works in the root directory of the evolutionaryscale repository."
        )
    return Path(
        mkimg(
            e="py,yaml,txt,toml,template,json,lock,sh,md",
            dest=str(path),
            force=[".env"],
            require_pixi=require_pixi,
        )
    )


def get_image_and_sbatch_dest(
    require_pixi: bool = True,
):
    image_dest = dump_code_image(require_pixi=require_pixi)
    sbatch_dest = image_dest.with_suffix("").with_suffix(".sh")
    return image_dest, sbatch_dest


class EvoSubmitContext:
    """Context manager for programmatically submitting jobs to SLURM.

    You first create a SubmitContext with a SubmissionConfig. This cfg
    specifies the array name, partition, and how many GPUs each job will
    use. Then, you can submit jobs with the submit method. The submit method
    takes in a config dataclass, which will be treated as the output of a
    python config.

    Currently only supports config dictionaries and dataclasses.

    The jobs will not be launched until the context exits.

    Usage example:
    ```
    cfg = SubmissionConfig("run1", "highpri", gpus=1)
    with EvoSubmitContext("hello.py", cfg) as ctx:
        ctx.submit(TestConfig(a=1, b=2))
        ctx.submit(TestConfig(a=1, b=2))
    ```

    """

    def __init__(
        self,
        file: str,
        submission_cfg: SubmissionConfig,
        verbose: bool = True,
        code_image_path: Path | None = None,
        force_array: bool = False,
        force_dict_cfg: bool = True,
    ):
        _verify_submission_cfg(submission_cfg)
        self.submission_cfg = submission_cfg
        self.verbose = verbose
        self.image_dest = code_image_path
        self.force_array = force_array
        self.force_dict_cfg = force_dict_cfg

        if file.endswith(".py"):
            self.file = file
        elif file in MAIN_FUNCTIONS:
            self.file = MAIN_FUNCTIONS[file]
        else:
            raise ValueError(f"Unknown main function {file}")

    def __enter__(self):
        if (
            "pytest" not in sys.modules
            and not Path("~/.aws/credentials").expanduser().exists()
        ):
            raise RuntimeError("Please set up your aws credentials")

        if self.image_dest is None:
            self.image_dest = dump_code_image()
        self.n_submissions = 0
        self.config_prefix = (
            f"{self.image_dest.with_suffix('').with_suffix('')}_{time.monotonic_ns()}"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.n_submissions == 0:
            print("No jobs submitted, skipping submission.")
            return

        assert self.image_dest is not None
        gpus = self.submission_cfg.gpus
        partition = self.submission_cfg.partition
        retry = self.submission_cfg.retry
        run_name = self.submission_cfg.run_name
        dry_run = self.submission_cfg.dry_run
        nice = self.submission_cfg.nice
        time_limit = self.submission_cfg.time_limit
        no_wandb = self.submission_cfg.no_wandb
        ray = self.submission_cfg.ray
        array_limit = self.submission_cfg.array_limit
        wait_until_completion = self.submission_cfg.wait_until_completion

        python_config_path = Path(self.config_prefix).with_suffix(".cfg.py")

        config_file_template = CONFIG_FILE_TEMPLATE
        if not self.force_dict_cfg:
            config_file_template = CONFIG_FILE_TEMPLATE_WITHOUT_DICT_CAST

        python_config_path.write_text(
            config_file_template.format(config_prefix=self.config_prefix)
        )

        if self.force_array:
            array = self.n_submissions
        else:
            if self.n_submissions > 1:
                array = self.n_submissions
            else:
                array = 0

        slurm_run(
            f"python {self.file} config={python_config_path}",
            self.image_dest,
            Path(self.config_prefix).with_suffix(".sh"),
            run_name=run_name,
            partition=partition,
            gpus=gpus,
            nice=nice,
            time_limit=time_limit,
            dry_run=dry_run,
            retry=retry,
            verbose=self.verbose,
            array=array,
            array_limit=array_limit,
            no_wandb=no_wandb,
            ray=ray,
            wait_until_completion=wait_until_completion,
        )

    def submit(self, config_to_submit):
        # Slurm array indices are 1-indexed, so we increment first.
        self.n_submissions += 1

        assert self.image_dest is not None
        file_prefix = Path(self.config_prefix + f"_{self.n_submissions}")

        config_pickle_path = file_prefix.with_suffix(".cfg.pkl")
        with open(config_pickle_path, "wb") as f:
            pickle.dump(config_to_submit, f)


__all__ = [
    "BASE_CONFIG_FILE_TEMPLATE",
    "CONFIG_FILE_TEMPLATE",
    "CONFIG_FILE_TEMPLATE_WITHOUT_DICT_CAST",
    "SBATCH",
    "SBATCH_RAY",
    "MAIN_FUNCTIONS",
    "_format_slurm_comment",
    "_is_training_job",
    "_guess_proj_from_command",
    "_slurm_job_id_string",
    "_slurm_retry_trap",
    "_maybe_path",
    "_get_user_jobs",
    "list_jobs_for_user",
    "_rerun_job",
    "_job_running",
    "get_wandb_run_name_from_script",
    "job_name_active",
    "current_job_partition",
    "rerun_job_by_id",
    "rerun_jobs_by_tag",
    "rerun_jobs_by_name",
    "slurm_run",
    "_verify_submission_cfg",
    "mkimg",
    "dump_code_image",
    "get_image_and_sbatch_dest",
    "SubmissionConfig",
    "EvoSubmitContext",
]
