import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

DIR_PATH = Path(__file__).parent
PROJECT_ROOT = DIR_PATH.parent

CONSOLE = Console()
BORDER_STYLE = "magenta"
ERROR = "red"


def run_isort(border_style: str = BORDER_STYLE) -> int:
    result = subprocess.run(["isort", "--check-only", PROJECT_ROOT], capture_output=True, text=True)
    output = result.stdout + result.stderr
    CONSOLE.print(
        Panel(
            output,
            title="isort --check-only",
            border_style=border_style if result.returncode == 0 else ERROR,
            title_align="left",
        ),
    )
    return result.returncode


def run_black(border_style: str = BORDER_STYLE) -> int:
    result = subprocess.run(["black", "--check", PROJECT_ROOT], capture_output=True, text=True)
    output = result.stdout + result.stderr
    CONSOLE.print(
        Panel(
            output,
            title="black --check",
            border_style=border_style if result.returncode == 0 else ERROR,
            title_align="left",
        ),
    )
    return result.returncode


def run_flake8(border_style: str = BORDER_STYLE) -> int:
    result = subprocess.run(["flake8", PROJECT_ROOT], capture_output=True, text=True)
    output = result.stdout + result.stderr
    CONSOLE.print(
        Panel(
            output,
            title="flake8",
            border_style=border_style if result.returncode == 0 else ERROR,
            title_align="left",
        ),
    )
    return result.returncode


def run_pyright(border_style: str = BORDER_STYLE) -> int:
    result = subprocess.run(["pyright", PROJECT_ROOT], capture_output=True, text=True)
    output = result.stdout + result.stderr
    CONSOLE.print(
        Panel(
            output,
            title="pyright",
            border_style=border_style if result.returncode == 0 else ERROR,
            title_align="left",
        ),
    )
    return result.returncode


def run_pytest(border_style: str = BORDER_STYLE) -> int:
    result = subprocess.run(["pytest", PROJECT_ROOT], capture_output=True, text=True)
    output = result.stdout + result.stderr
    CONSOLE.print(
        Panel(
            output,
            title="pytest",
            border_style=border_style if result.returncode == 0 else ERROR,
            title_align="left",
        ),
    )
    return result.returncode


def main():
    for check in [run_isort, run_black, run_flake8, run_pyright, run_pytest]:
        returncode = check()
        if returncode != 0:
            return returncode


if __name__ == "__main__":
    main()
