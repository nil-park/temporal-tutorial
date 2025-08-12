import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .run_all_tests import run_flake8, run_pyright

DIR_PATH = Path(__file__).parent
PROJECT_ROOT = DIR_PATH.parent

CONSOLE = Console()
BORDER_STYLE = "cyan"


def run_isort():
    result = subprocess.run(["isort", PROJECT_ROOT], capture_output=True, text=True)
    output = result.stdout + result.stderr
    CONSOLE.print(
        Panel(
            output,
            title="isort",
            border_style=BORDER_STYLE,
            title_align="left",
        ),
    )


def run_black():
    result = subprocess.run(["black", PROJECT_ROOT], capture_output=True, text=True)
    output = result.stdout + result.stderr
    CONSOLE.print(
        Panel(
            output,
            title="black",
            border_style=BORDER_STYLE,
            title_align="left",
        ),
    )


def main():
    run_isort()
    run_black()
    run_flake8(BORDER_STYLE)
    run_pyright(BORDER_STYLE)


if __name__ == "__main__":
    main()
