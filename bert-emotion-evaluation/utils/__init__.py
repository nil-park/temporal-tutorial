from colorama import Fore, Style


def print_cyan(text: str) -> None:
    print(Fore.CYAN + text + Style.RESET_ALL)


def print_magenta(text: str) -> None:
    print(Fore.MAGENTA + text + Style.RESET_ALL)


def print_yellow(text: str) -> None:
    print(Fore.YELLOW + text + Style.RESET_ALL)


def print_green(text: str) -> None:
    print(Fore.GREEN + text + Style.RESET_ALL)


def print_red(text: str) -> None:
    print(Fore.RED + text + Style.RESET_ALL)
