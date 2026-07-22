#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for dealing with files"""

import os
import glob

from .decorators import deprecate_positional_args


__all__ = [
    "find_files",
]

# --- christiangeorgelucas/audio-tools patch -------------------------------
# Upstream librosa.util.files also defines example()/ex()/list_examples()/
# example_info(), which fetch bundled demo recordings over HTTPS via the
# `pooch` package. audio-tools is offline/deterministic (no network calls)
# and never uses those functions, so this vendored copy drops them along
# with their `pooch` and `pkg_resources` imports entirely — pooch's own
# declared dependency on `requests` (which pulls in `certifi`, MPL-2.0)
# would otherwise be forced into the installed dependency closure even
# though it is never exercised. See vendor/librosa/LICENSE.md (ISC) for
# the original librosa license; this file remains ISC-licensed librosa
# source with functions removed, not rewritten.
# ---------------------------------------------------------------------------


@deprecate_positional_args
def find_files(
    directory, *, ext=None, recurse=True, case_sensitive=False, limit=None, offset=0
):
    """Get a sorted list of (audio) files in a directory or directory sub-tree.

    Examples
    --------
    >>> # Get all audio files in a directory sub-tree
    >>> files = librosa.util.find_files('~/Music')

    >>> # Look only within a specific directory, not the sub-tree
    >>> files = librosa.util.find_files('~/Music', recurse=False)

    >>> # Only look for mp3 files
    >>> files = librosa.util.find_files('~/Music', ext='mp3')

    >>> # Or just mp3 and ogg
    >>> files = librosa.util.find_files('~/Music', ext=['mp3', 'ogg'])

    >>> # Only get the first 10 files
    >>> files = librosa.util.find_files('~/Music', limit=10)

    >>> # Or last 10 files
    >>> files = librosa.util.find_files('~/Music', offset=-10)

    >>> # Avoid including search patterns in the path string
    >>> import glob
    >>> directory = '~/[202206] Music'
    >>> directory = glob.escape(directory)  # Escape the special characters
    >>> files = librosa.util.find_files(directory)

    Parameters
    ----------
    directory : str
        Path to look for files

    ext : str or list of str
        A file extension or list of file extensions to include in the search.

        Default: ``['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']``

    recurse : boolean
        If ``True``, then all subfolders of ``directory`` will be searched.

        Otherwise, only ``directory`` will be searched.

    case_sensitive : boolean
        If ``False``, files matching upper-case version of
        extensions will be included.

    limit : int > 0 or None
        Return at most ``limit`` files. If ``None``, all files are returned.

    offset : int
        Return files starting at ``offset`` within the list.

        Use negative values to offset from the end of the list.

    Returns
    -------
    files : list of str
        The list of audio files.
    """

    if ext is None:
        ext = ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]

    elif isinstance(ext, str):
        ext = [ext]

    # Cast into a set
    ext = set(ext)

    # Generate upper-case versions
    if not case_sensitive:
        # Force to lower-case
        ext = set([e.lower() for e in ext])
        # Add in upper-case versions
        ext |= set([e.upper() for e in ext])

    files = set()

    if recurse:
        for walk in os.walk(directory):
            files |= __get_files(walk[0], ext)
    else:
        files = __get_files(directory, ext)

    files = list(files)
    files.sort()
    files = files[offset:]
    if limit is not None:
        files = files[:limit]

    return files


def __get_files(dir_name, extensions):
    """Helper function to get files in a single directory"""

    # Expand out the directory
    dir_name = os.path.abspath(os.path.expanduser(dir_name))

    myfiles = set()

    for sub_ext in extensions:
        globstr = os.path.join(dir_name, "*" + os.path.extsep + sub_ext)
        myfiles |= set(glob.glob(globstr))

    return myfiles
