import shlex

import click

from .submit import (
    get_image_and_sbatch_dest,
    job_name_active,
    rerun_job_by_id,
    rerun_jobs_by_name,
    rerun_jobs_by_tag,
    slurm_run,
)


@click.command(
    help="""Examples:

\b
Simple command to run a training run:
    slurm_run --gpus 8 --run-name MY_RUN --retry -- python train.py...

\b
Generate bash scripts and print it out, but don't run things:
    slurm_run --gpus 8 --run-name MY_RUN --dry-run --retry -- python train.py...

\b
Run an array job - we don't allow envvars for security purposes, please detect SLURM_ARRAY_TASK_ID in your script:
    slurm_run --gpus 8 --run-name MY_RUN --array 8 -- python train.py...
"""
)
@click.argument("command", nargs=-1)
@click.option(
    "--run-name",
    type=str,
    default=None,
    help="The (optional) name for the run. Will also override logger.run_name if applicable.",
)
@click.option(
    "--partition",
    type=str,
    default="midpri",
    help="The partition to use, defaults to all on slurm.",
)
@click.option("--gpus", type=int, default=8, help="The number of GPUs requested.")
@click.option("--cpus", type=int, default=8, help="Number of CPUs per task.")
@click.option(
    "--ntasks",
    type=int,
    default=None,
    help="Number of tasks to run for a CPU-only job.",
)
@click.option("--nice", type=int, default=0, help="The nice level of the job.")
@click.option(
    "--time-limit", type=str, default="7-0", help="The time limit of the job."
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="If set to True, it will only print the script to execute.",
)
@click.option(
    "--retry",
    is_flag=True,
    default=False,
    help="Only for slurm, it can requeue jobs on preempt / signal.",
)
@click.option(
    "--array",
    type=int,
    default=0,
    help="Only for slurm, it can optionally launch an array job when array > 0.",
)
@click.option(
    "--array-limit",
    type=int,
    default=None,
    help="Only for slurm, if set, this will restrict the max number of running array jobs.",
)
@click.option(
    "--no-wandb",
    is_flag=True,
    default=False,
    help="If set to True, it will not export data to Weights and Biases.",
)
@click.option(
    "--tag",
    type=str,
    default=None,
    help="If set, will tag the job with the specified tag",
)
@click.option(
    "--rerun-id",
    type=int,
    default=None,
    help="If set, will rerun the job with the specified job id or array id.",
)
@click.option(
    "--rerun-tag",
    type=str,
    default=None,
    help="If set, will rerun the job(s) with the specified tag.",
)
@click.option(
    "--rerun-name",
    type=str,
    default=None,
    help="If set, will rerun the last job(s) with the specified name.",
)
@click.option("--ray", is_flag=True, help="If set, will run using ray-on-slurm")
@click.option(
    "--exclusive",
    is_flag=True,
    default=False,
    help="If set, will run the job with exclusive access to the nodes.",
)
@click.option(
    "--dependency",
    type=str,
    default="",
    help="Slurm job dependency. For details: https://slurm.schedmd.com/sbatch.html#OPT_dependency",
)
@click.option(
    "--notify",
    is_flag=True,
    default=False,
    help="whether to notify the user via slack or not",
)
@click.option(
    "--venv",
    type=click.Choice(["pixi", "uv"], case_sensitive=False),
    help="If pixi, will snapshot and run commands under the current pixi env.",
)
def main(
    command,
    run_name,
    partition,
    gpus,
    cpus,
    ntasks,
    nice,
    time_limit,
    dry_run,
    retry,
    array,
    array_limit,
    no_wandb,
    tag,
    rerun_id,
    rerun_tag,
    rerun_name,
    ray,
    exclusive,
    dependency,
    notify,
    venv,
):
    # Rerun existing jobs
    if rerun_id:
        rerun_job_by_id(rerun_id)
        return 0
    if rerun_tag:
        rerun_jobs_by_tag(rerun_tag)
        return 0
    if rerun_name:
        rerun_jobs_by_name(rerun_name)
        return 0

    if partition not in ["highpri", "midpri", "lowpri", "hero", "antihero"]:
        raise ValueError(
            f"Specified partition ({partition}) not in (high|mid|low)pri, hero, or "
            f"antihero."
        )

    if run_name and job_name_active(run_name):
        # Avoid collisions with other active jobs
        raise ValueError(
            f"Job with name {run_name} is already active, please select another name."
        )

    require_pixi = (venv.lower() == "pixi")
    image_dest, sbatch_dest = get_image_and_sbatch_dest(require_pixi=require_pixi)

    command = " ".join(shlex.quote(x) for x in command)

    _ = slurm_run(
        command=command,
        image=image_dest,
        sbatch_dest=sbatch_dest,
        run_name=run_name,
        partition=partition,
        gpus=gpus,
        cpus=cpus,
        ntasks=ntasks,
        nice=nice,
        time_limit=time_limit,
        dry_run=dry_run,
        retry=retry,
        array=array,
        array_limit=array_limit,
        verbose=True,
        no_wandb=no_wandb,
        tag=tag,
        ray=ray,
        exclusive=exclusive,
        dependency=dependency,
        comment=None,
        notify=notify,
        venv=venv,
    )
    return 0
