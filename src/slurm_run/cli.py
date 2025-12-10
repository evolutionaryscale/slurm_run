import click
from tabulate import tabulate

from .slurm import main as slurm_submit
from .submit import list_jobs_for_user


@click.group()
def main():
    pass


main.add_command(slurm_submit, name="submit")


@main.command()
@click.option(
    "--starttime",
    type=str,
    default="now-1days",
    help="Slurm start time used for filtering jobs. Defaults to now-1days.",
)
@click.option(
    "--user",
    type=str,
    default=None,
    help="If set, will show all jobs for the specified user, else defaults to current user.",
)
@click.option(
    "--columns",
    type=str,
    default="JobID,JobName,Partition,State,Elapsed,Stdout",
    help="Comma-separated list of columns to display. Default: JobID,JobName,Partition,State,Elapsed. Available: JobID,ArrayJobID,JobName,Tag,Partition,State,Elapsed,Command,Stdout,Stderr,WandBJobName",
)
def jobs(starttime, user, columns):  # Dump job info for user to stdout only
    job_data = list_jobs_for_user(starttime=starttime, user=user, columns=columns)
    print(tabulate(job_data, headers="keys", tablefmt="pretty"))
    return 0


if __name__ == "__main__":
    main()
