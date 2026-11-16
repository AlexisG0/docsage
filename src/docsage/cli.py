"""Command line entry point."""

from __future__ import annotations

import typer

from docsage import __version__

app = typer.Typer(help="Ask questions about your technical docs.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Keep subcommands addressable by name even while there is only one."""


@app.command()
def version() -> None:
    """Print the installed docsage version."""
    typer.echo(__version__)
