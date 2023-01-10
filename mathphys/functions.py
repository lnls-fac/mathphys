"""Useful functions."""
import os as _os
from collections import namedtuple as _namedtuple
from functools import partial as _partial
import pickle as _pickle

import numpy as _np
from git import Repo as _Repo


def generate_random_numbers(n_part, dist_type='exp', cutoff=3):
    """Generate random numbers with a cutted off dist_type distribution.

    Inputs:
        n_part = size of the array with random numbers
        dist_type = assume values 'exponential', 'normal' or 'uniform'.
        cutoff = where to cut the distribution tail.
    """
    dist_type = dist_type.lower()
    if dist_type in 'exponential':
        func = _partial(_np.random.exponential, 1)
    elif dist_type in 'normal':
        func = _np.random.randn
    elif dist_type in 'uniform':
        func = _np.random.rand
    else:
        raise NotImplementedError('Distribution type not implemented yet.')

    numbers = func(n_part)
    above, *_ = _np.asarray(_np.abs(numbers) > cutoff).nonzero()
    while above.size:
        parts = func(above.size)
        indcs = _np.abs(parts) > cutoff
        numbers[above[~indcs]] = parts[~indcs]
        above = above[indcs]

    if dist_type in 'uniform':
        numbers -= 1/2
        numbers *= 2
    return numbers


def get_namedtuple(name, field_names, values=None):
    """Return an instance of a namedtuple Class.

    Inputs:
        - name:  Defines the name of the Class (str).
        - field_names:  Defines the field names of the Class (iterable).
        - values (optional): Defines field values . If not given, the value of
            each field will be its index in 'field_names' (iterable).

    Raises ValueError if at least one of the field names are invalid.
    Raises TypeError when len(values) != len(field_names)
    """
    if values is None:
        values = range(len(field_names))
    field_names = [f.replace(' ', '_') for f in field_names]
    return _namedtuple(name, field_names)(*values)


def save_pickle(data, fname, overwrite=False):
    """Save data to file in pickle format.

    Inputs:
        data - python object to be saved.
        fname - name of the file to be saved. With or without ".pickle"."
        overwrite - whether to overwrite existing file (Optional).

    Raises `FileExistsError` in case `overwrite` is `False` and file exists.
    """
    if not fname.endswith('.pickle'):
        fname += '.pickle'
    if not overwrite and _os.path.isfile(fname):
        raise FileExistsError(f'file {fname} already exists.')
    with open(fname, 'wb') as fil:
        _pickle.dump(data, fil)


def load_pickle(fname):
    """Load ".pickle" file.

    Inputs:
        fname - filename. May or may not contain the ".pickle" extension.

    Outputs:
        data - content of file as a python object.
    """
    if not fname.endswith('.pickle'):
        fname += '.pickle'
    with open(fname, 'rb') as fil:
        data = _pickle.load(fil)
    return data


def repository_info(repo_path):
    """Get repository information.

    Args:
        repo_path (str): Repository path.

    Returns:
        repo_info (dict): Repository info.
            path, active_branch, last_tag, last_commit, is_dirty.

    """
    repo_info = {}
    try:
        repo = _Repo(repo_path, search_parent_directories=True)
    except:
        print(f'Repository path not found: {repo_path:s}.')
        return repo_info
    branch = repo.active_branch.name
    last_tag = repo.tags[-1].name
    last_commit = repo.head.commit.hexsha[:7]
    is_dirty = repo.is_dirty()
    repo_info['path'] = repo_path
    repo_info['active_branch'] = branch
    repo_info['last_tag'] = last_tag
    repo_info['last_commit'] = last_commit
    repo_info['is_dirty'] = is_dirty
    return repo_info
