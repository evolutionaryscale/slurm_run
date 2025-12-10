import click

from .submit import submit
from .jobs import jobs

@click.group()
def main():
    pass


main.add_command(submit, name="submit")
main.add_command(jobs, name="jobs")


if __name__ == "__main__":
    main()
