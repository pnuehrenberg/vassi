from pathlib import Path
from typing import override

import yaml


class _NoAliasDumper(yaml.SafeDumper):
    """
    Helper class to dump yaml without aliases.
    """

    @override
    def ignore_aliases(self, data: ...) -> bool:
        return True


def _construct_yaml_tuple(self: yaml.BaseLoader, node: yaml.SequenceNode):
    """
    Helper function to construct a tuple from a YAML sequence.
    """
    return tuple(self.construct_sequence(node))


class _TupleLoader(yaml.SafeLoader):
    """
    Helper class to load all sequences in YAML as tuples.
    """

    pass


_TupleLoader.add_constructor("tag:yaml.org,2002:seq", _construct_yaml_tuple)


def to_yaml(dump: ..., *, file_name: str | Path) -> None:
    """
    Helper function to write an object to a YAML file.

    Parameters:
        dump: The object to be dumped.
        file_name: The name of the file to write the YAML to.
    """
    with open(file_name, "w") as yaml_file:
        _ = yaml_file.write(yaml.dump(dump, Dumper=_NoAliasDumper, sort_keys=False))


def from_yaml(file_name: str | Path) -> object:
    """
    Helper function to read an object from a YAML file.

    Note that all lists are loaded as tuples.

    Parameters:
        file_name: The name of the file to read the YAML from.
    """
    with open(file_name, "r") as yaml_file:
        return yaml.load(yaml_file, Loader=_TupleLoader)
