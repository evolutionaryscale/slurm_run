import os
import subprocess
from contextlib import contextmanager


def test_if_pixi_lock_file_is_valid() -> bool:
    """
    Test if the pixi lock file is valid by running a simple command.

    NOTE(rverkuil): There may be a more straightforward way to do this.
    """
    p = subprocess.run("pixi run --locked echo".split(), capture_output=True)
    return p.returncode == 0


@contextmanager
def temporarily_unset_slurm_env_vars(env_var_substring: str = "SLURM"):
    """Temporarily unset all environment variables containing `env_var_substring`.
     At a minimum this substring should include "SLURM".

    NOTE(rverkuil): This is necessary to safely launch work from
    inside of an existing, (e.g. an interactive) job"""

    assert "SLURM" in env_var_substring, (
        "For safety, we assert that the env var being disabled contains subtring SLURM. "
        "This is true of all SLURM env vars."
    )

    deleted = {}
    for k, v in os.environ.items():
        if env_var_substring in k:
            deleted[k] = os.environ.pop(k)
    yield
    for k, v in deleted.items():
        os.environ[k] = v
