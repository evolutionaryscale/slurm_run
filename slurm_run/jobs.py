from datetime import timedelta
import os
import re

import click
from tabulate import tabulate

from .utils import get_user_jobs


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


def list_jobs_for_user(
    starttime: str,
    user: str | None = None,
    columns: str = "JobID,JobName,Partition,State,Elapsed",
) -> list[dict]:
    """Dict of selected job info for the current user"""
    user_jobs = get_user_jobs(starttime, user)
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
