"""Filesystem paths that retain Windows long-path support without OS changes."""
import os
from pathlib import Path


def filesystem_path(path):
    path=Path(path)
    if os.name!='nt': return path
    value=str(path.absolute())
    if value.startswith('\\\\?\\'): return path
    if value.startswith('\\\\'): return Path('\\\\?\\UNC\\'+value[2:])
    return Path('\\\\?\\'+value)


def canonical_path(path):
    """Keep namespace prefixes out of checkpoint identity/provenance labels."""
    value=str(path)
    if value.startswith('\\\\?\\UNC\\'): return Path('\\\\'+value[8:])
    if value.startswith('\\\\?\\'): return Path(value[4:])
    return Path(value)
