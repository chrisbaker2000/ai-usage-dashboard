"""A stand-in for ``subprocess.run`` for code that drives system tools and vendor CLIs."""

import subprocess
from typing import Any


class FakeRun:
    """Stands in for ``subprocess.run`` and records every command it was asked to run."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "", error: Exception | None = None):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.commands: list[list[str]] = []
        self.returncodes: dict[str, int] = {}
        self.sequences: dict[str, list[int]] = {}

    def __call__(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        """Answer with the subcommand's next scripted status, then its fixed status, then the default."""
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        subcommand = command[1] if len(command) > 1 else ""
        scripted = self.sequences.get(subcommand)
        returncode = scripted.pop(0) if scripted else self.returncodes.get(subcommand, self.returncode)
        return subprocess.CompletedProcess(command, returncode, self.stdout, self.stderr)

    def subcommands(self) -> list[str]:
        return [command[1] for command in self.commands]
