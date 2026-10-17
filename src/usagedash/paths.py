import os
from pathlib import Path

from usagedash.errors import ConfigError


def directory_identity(path: Path) -> Path:
    try:
        resolved = path.resolve()
        return _spelling_stored_for(resolved)
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot identify {path}: {type(error).__name__}") from error


def _spelling_stored_for(resolved: Path) -> Path:
    current = Path(resolved.anchor)
    for name in resolved.relative_to(current).parts:
        current = _child_stored_as(current, name)
    return current


def _child_stored_as(parent: Path, name: str) -> Path:
    given = parent / name
    if not parent.is_dir():
        return given
    folded: str | None = None
    with os.scandir(parent) as entries:
        for entry in entries:
            if entry.name == name:
                return parent / entry.name
            if folded is None and entry.name.lower() == name.lower():
                folded = entry.name
    if folded is None:
        return given
    stored = parent / folded
    if given.exists() and os.path.samefile(given, stored):
        return stored
    return given


def same_directory(first: Path, second: Path) -> bool:
    """Raises:
    ConfigError: if either path cannot be examined for any other reason."""
    try:
        return os.path.samefile(first, second)
    except (FileNotFoundError, NotADirectoryError):
        pass
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot tell whether {first} is {second}: {type(error).__name__}") from error
    try:
        if first.exists() != second.exists():
            return False
        return first.resolve() == second.resolve()
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot tell whether {first} is {second}: {type(error).__name__}") from error
