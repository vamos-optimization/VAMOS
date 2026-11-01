from __future__ import annotations

import argparse
import re
from typing import Any, NoReturn

_SPEC_KEY_ATTR = "_vamos_spec_key"


class SpecArgumentParser(argparse.ArgumentParser):
    """Argument parser whose errors also name the config-file key that set an offending option.

    ``config_keys`` maps an option string (``--max-evaluations``) to the config key that gave
    its value (``defaults.max_evaluations``); it stays empty without a config file.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config_keys: dict[str, str] = {}

    def error(self, message: str) -> NoReturn:
        flags = dict.fromkeys(re.findall(r"--[a-z0-9][a-z0-9-]*", message))
        keys = [self.config_keys[flag] for flag in flags if flag in self.config_keys]
        if keys:
            message = f"{message} (set by {', '.join(keys)} in the config file)"
        super().error(message)


def add_spec_argument(parser: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> argparse.Action:
    """
    Register an argparse argument that can also be set via experiment spec defaults.

    The spec key is the argument's `dest` name.
    """
    action = parser.add_argument(*args, **kwargs)
    setattr(action, _SPEC_KEY_ATTR, action.dest)
    return action


def parser_spec_keys(parser: argparse.ArgumentParser) -> set[str]:
    """Return all spec-backed keys registered on this parser."""
    keys: set[str] = set()
    for action in parser._actions:  # argparse stores actions internally
        key = getattr(action, _SPEC_KEY_ATTR, None)
        if isinstance(key, str) and key:
            keys.add(key)
    return keys
