import click

from .submit import (
    _format_slurm_comment,
    get_image_and_sbatch_dest,
    job_name_active,
    slurm_run,
)


@click.command(
    help="""Examples:
\b
Run nightly tests - all tests, training tests only, or inference tests only:
    evorun test all --run-name TEST_RUN
    evorun test training --run-name TEST_RUN
    evorun test inference --run-name TEST_RUN
"""
)
@click.argument("tests", default="all")
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
    "--tag",
    type=str,
    default=None,
    help="If set, will tag the job with the specified tag",
)
@click.option(
    "--test-filter",
    type=str,
    default=None,
    help="If set, will run only tests with the specified filter.",
)
@click.option(
    "--dependency",
    type=str,
    default="",
    help="Slurm job dependency. For details: https://slurm.schedmd.com/sbatch.html#OPT_dependency",
)
@click.option(
    "--constraint",
    type=click.Choice(["h100-reserved", "h200-reserved"]),
    default="h100-reserved",
    help="Select between different node types",
)
@click.option(
    "--notify",
    is_flag=True,
    default=False,
    help="whether to notify the user via slack or not",
)
def main(
    tests,
    run_name,
    partition,
    nice,
    time_limit,
    dry_run,
    retry,
    tag,
    test_filter,
    dependency,
    constraint,
    notify,
):
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

    image_dest, sbatch_dest = get_image_and_sbatch_dest()

    # TODO(jmaccarl): Expand this functionality as needed to support more customizations.
    # For now we only support running tests by tag, but could support by GPU, etc.
    num_gpus = 4  # Should match MAX_GPU_REQUIREMENT in tests/nightly_test_wrapper.py
    print(f"Running tests... overriding the number of GPUs to {num_gpus}")

    # 5 workers. 2 CPU, 2 1-GPU, 1 2-GPUs.
    # Also impose 10-min timeout. Any single nightly test shouldn't take this long.
    # Timeout usually indicates CUDA or torch errors that are not captured.
    common_cmd = "pytest --import-mode=importlib --no-cov -vvv -n5 --dist=loadgroup --timeout=600"
    if tests == "training":
        test_command = f"{common_cmd} -m training --max-gpus={num_gpus}"
    elif tests == "inference":
        test_command = (
            f"{common_cmd} -m 'nightly and not training' --max-gpus={num_gpus}"
        )
    elif tests == "all":
        test_command = f"{common_cmd} -m nightly --max-gpus={num_gpus}"
    else:
        raise ValueError(
            f"Invalid argument: {tests}. Must be one of: [training, inference, all]."
        )

    if not run_name:
        run_name = f"test-{tests}"

    if test_filter:
        test_command += f' -k "{test_filter}"'

    _ = slurm_run(
        test_command,
        image=image_dest,
        sbatch_dest=sbatch_dest,
        run_name=run_name,
        partition=partition,
        gpus=num_gpus,
        nice=nice,
        time_limit=time_limit,
        dry_run=dry_run,
        retry=retry,
        array=False,
        array_limit=False,
        verbose=True,
        no_wandb=True,  # always disable wandb for tests
        tag=tag,
        ray=False,  # not supported right now
        exclusive=False,  # since # GPUs < 8, never use exclusive mode
        no_srun=True,  # Important: tests will launch their own processes, so we don't need to use srun
        dependency=dependency,
        constraint=constraint,
        comment=_format_slurm_comment(job_type="nightly", proj="ci"),
        notify=notify,
    )
    return 0
