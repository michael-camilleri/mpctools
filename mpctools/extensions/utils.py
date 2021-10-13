"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""

from datetime import timedelta, datetime
from itertools import tee, islice
import numpy as np
import contextlib
import shutil
import copy
import os


################################################
#            Collections Processing            #
################################################


class BitField:
    """
    A Class to define a bit-field
    """
    def __init__(self, width, vals=()):
        """
        Initialise with the list of values set to True and a specified width.
        :param width: The size of the flag
        :param vals: The set of 'on' values
        """
        if len(vals) > 0:
            self.__flag = f"{(10 ** (width - 1 - np.asarray(vals, dtype=int))).sum():0{width}d}"
        else:
            self.__flag = "0" * width
        self.__it = None

    def __repr__(self):
        return self.__flag

    def ison(self, item):
        return self.__flag[item] == "1"

    def any(self):
        return "1" in self.__flag

    def __contains__(self, item):
        return self.ison(item)

    def __len__(self):
        return len(self.__flag)

    def __iter__(self):
        self.__it = 0
        return self

    def __next__(self):
        idx = self.__flag.find('1', self.__it)
        if idx > -1:
            self.__it = idx+1
            return idx
        else:
            self.__it = -1
            raise StopIteration


class Pool:
    """
    Defines a pool of object assignments. By this we mean that it keeps track of a property for a
    set of indices (hashables), and when indices die, can assign to new ones.
    """

    def __init__(self, values):
        """
        Creates the Pool object

        :param values: The allowable values that can be returned. Must be an indexable iterable.
        """
        self.__values = values
        self.__val_map = {}
        self.__unused_val = {*range(len(values))}

    def update(self, indices):
        """
        Updates the state with the new index-set

        :param indices: New set of indices. This can be anything that is hashable.
        :return: self, for chaining.
        """
        for _t_id in set(self.__val_map.keys()).difference(set(indices)):
            self.__unused_val.add(self.__val_map.pop(_t_id))
        for _t_id in set(indices).difference(set(self.__val_map.keys())):
            self.__val_map[_t_id] = self.__unused_val.pop()
        return self

    def __getitem__(self, item):
        return self.__values[self.__val_map[item]]

    @property
    def capacity(self):
        return len(self.__values)

    @property
    def values(self):
        return self.__values

    def assigned(self):
        """
        Returns the assignment status of each value (much like a reverse dict).

        :return: A Dictionary showing for each value the assigned index or None As per Python
        Standard, the returned order follows the set of values as passed to the Initialiser.
        """
        reverse_dict = {v: None for v in self.__values}
        for index, val_idx in self.__val_map.items():
            reverse_dict[self.__values[val_idx]] = index
        return reverse_dict


def window(iterable, size):
    """
    Return a sliding window iterator over elements.

    From https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator

    :param iterable: An iterable
    :param size:     Size of the sliding window
    :return:         Iterator
    """
    itrs = tee(iterable, size)
    shiftedStarts = [islice(anItr, s, None) for s, anItr in enumerate(itrs)]
    return zip(*shiftedStarts)


def to_tuple(value):
    """
    Static (Module) Method for converting a scalar into a tuple if not already (tuple/list/numpy
    array). Will also ignore None. Note, that this does not handle dict types!

    :param value: The Value to convert
    :return: A Tuple containing value if a scalar, or value if already a list/tuple/numpy array/none
    """
    return value if (type(value) in (tuple, list, np.ndarray) or value is None) else (value,)


def to_list(value):
    """
    Static (Module) Method for converting a scalar into a list if is not already an iterable
    (tuple/list/array)

    :param value: The Value to convert
    :return: A List containing value if a scalar, or value if already a list/tuple/numpy array/none
    """
    return (
        value if (type(value) in (tuple, list, np.ndarray) or value is None) else [value,]
    )


def to_scalar(value):
    """
    Converts a single-element tuple into a scalar value if not already. To do this, it attempts to
    dereference the first element if it is of type tuple, list or np.ndarray

    :param value:
    :return:
    """
    return value[0] if (type(value) in (tuple, list, np.ndarray)) else value


def extend_dict(_d1, _d2, deep=False):
    """
    This function may be used to extend the 'list/array'-type elements of _d1 by corresponding
    entries in _d2. The elements in either case must support the extend method (i.e. are typically
    lists) - scalars are however supported through the to_list method. Note that if a key exists
    in _d2 and not in _d1, it is automatically created as a list.

    :param _d1: Dictionary to extend: will be modified
    :param _d2: Dictionary to copy data from.
    :param deep: If True, then the elements in _d2 are deep copied when extending/creating
    :return:    Updated _d1 for chaining etc...
    """
    # Extend/Copy values
    for key, value in _d2.items():
        if key in _d1:
            _d1[key].extend(to_list(copy.deepcopy(value) if deep else value))
        else:
            _d1[key] = to_list(copy.deepcopy(value) if deep else value)

    # Return _d1
    return _d1


def dzip(*_dcts):
    """
    Generate an iterator over two (or more) dictionaries, parsing only the common keys.

    :param _dcts:   Iterable of dictionaries
    :return:        Enumerated iteration over common elements of dictionaries
    """
    for i in sorted(set(_dcts[0]).intersection(*_dcts[1:])):
        yield i, tuple(d[i] for d in _dcts)


def dict_invert(_dict):
    """
    Inverts the Key-Value pairs in a dictionary.
    ***IMP***: They Value entries themselves must be unique
    The function also works on lists (putting values as keys and the index in the list as values

    :param _dict: The dictionary to invert or a list to extract from
    :return: A new dict object which the key-value entries reversed (i.e. values become keys and
             v.v.)
    """
    if type(_dict) is dict:
        return {v: k for k, v in _dict.items()}
    else:
        return {v: k for k, v in enumerate(_dict)}


def glen(generator):
    """
    Compute the Length of a generator expression:

    This will basically execute the generator, so may have side-effects!

    :param generator: Generator to run
    :return:
    """
    return sum(1 for _ in generator)


def pad(a, length, value):
    """
    Pad a list (not tuple). Note that if length is less than the len(a) then this has no effect.

    :param a:       The original List. Must be a list and NOT a numpy array!
    :param length:  The desired length
    :param value:   The Value to fill with
    :return:        Padded List. A copy is always returned!
    """
    if length > len(a):
        return a + [value] * (length - len(a))
    else:
        return copy.copy(a)


def masked_list(l, m, mask_in=True):
    """
    Returns a copy of a list in which the elements in list are masked in/out as None

    :param l: List to operate on. Will be copied
    :param m: The indices of the elements to mask in/out
    :param mask_in: If True (default), then elements **NOT** in m will be masked out: otherwise,
              elements in m will be masked out.
    :return:  Copy of the list with appropriate elements masked out.
    """
    if mask_in:
        lcpy = [None for _ in l]
        for e in m:
            lcpy[e] = l[e]
    else:
        lcpy = copy.deepcopy(l)
        for e in m:
            lcpy[e] = None
    return lcpy


def default(var, val):
    """
    Replaces var with val if var is None:

    :param var: The variable to handle
    :param val: The value to use if var is None
    :return: var if var is not None, else val
    """
    return var if var is not None else val


################################################
#            Printing & Formatting             #
################################################


class NullableSink:
    """
    Defines a wrapper class which supports a nullable initialisation
    """

    def __init__(self, obj=None):
        self.Obj = obj

    def write(self, *args):
        if self.Obj is not None:
            self.Obj.write(*args)

    def flush(self):
        if self.Obj is not None:
            self.Obj.flush()

    def print(self, *args):
        self.write(*args)
        self.write("\n")
        self.flush()


def name(obj):
    """
    Static (Module) Method for outputting the name of a data type. This amounts to calling the
    __name__ method on the passed obj (or its class type) if it exists, or calling string
    otherwise on it... (mainly for None Types)

    :param obj: An instance or class type.
    :return: string representation of 'obj'
    """
    if isinstance(obj, type):
        if hasattr(obj, "__name__"):
            return obj.__name__
        else:
            return str(obj)
    else:
        return obj.__class__.__name__


def float_list(_list, prec=2):
    """
    Print a List of Floating point numbers with arbitrary precision

    :param _list: List of floats
    :param prec:  Precision
    :return:
    """
    return ["{1:.{0}f}".format(prec, f) for f in _list]


def time_list(_list, fmt="%H:%M:%S"):
    """
    Formats a list of time-points in 'HH:MM:SS'. Currently, only supports _list values in ms

    :param _list:
    :param fmt:  String format to employ
    :return: String representation
    """
    s_l = []
    for t in _list:
        s_l.append((datetime(1970, 1, 1) + timedelta(milliseconds=t)).strftime(fmt))
    return s_l


def show_time(_time, minimal=True, ms=False):
    """
    Shows a time-point in the most appropriate manner (Days/Hours/Minutes/Seconds)

    :param _time: Time (in seconds)
    :param minimal: If True, and less than 60, show in minimal view (s.ms). Does not effect if
                    more than 1 minute.
    :param ms: If True, show millisecond precision
    :return: String representation
    """
    if _time < 60 and minimal:
        return f"{_time:.3f}s" if ms else f"{int(_time)}s"
    else:
        _time_str = datetime(1970, 1, 1) + timedelta(seconds=float(_time))
        if _time < 3600:
            return _time_str.strftime("%M:%S.%f")[:-3] if ms else _time_str.strftime("%M:%S")
        elif _time < 86400:
            return _time_str.strftime("%H:%M:%S.%f")[:-3] if ms else _time_str.strftime("%H:%M:%S")
        else:
            _time_str -= timedelta(hours=24)
            return (
                _time_str.strftime("%jD+%H:%M:%S.%f")[:-3]
                if ms
                else _time_str.strftime("%jD+%H:%M:%S")
            )


def int_list(_list, _sort=True):
    """
    Formats a list of unique integers, summarising ranges
    :param _list: List to display
    :param _sort: If True, sort the elements
    :return:
    """
    _list = np.sort(_list).astype(int) if _sort else np.asarray(_list, dtype=int)
    _diff = [*(np.diff(_list) > 1), True]

    st = _list[0]
    ranges = []
    for el in np.argwhere(_diff):
        nd = _list[el][0]
        if st == nd:
            ranges.append(f"{st}")
        elif nd == st + 1:
            ranges.append(f"{st}")
            ranges.append(f"{nd}")
        else:
            ranges.append(f"{st}-{nd}")
        st = _list[el + 1][0] if el < len(_list) - 1 else None

    return ", ".join(ranges)


def str_width(_iterable):
    """
    Returns the maximum width of all strings in iterable

    :param _iterable:
    :return:
    """
    return max([len(str(s)) for s in _iterable])


def dict_width(_dict):
    """
    Returns the maximum length of all elements in a dictionary. Note that if the dict contains
    strings, strwidth should be used, as this will always return 1.

    :param _dict: Dictionary: elements may or may not implement the len function
    :return:
    """
    return max([len(to_tuple(s)) for s in _dict.values()])


def short_int(_int):
    """
    Returns a shorted representation of large numbers

    :param _int: Integer (or float) value
    :return:     String representation
    """
    if _int < 1000:
        return str(_int)
    elif _int < 1000000:
        return "{0:.4g}K".format(_int / 1000)
    elif _int < 1000000000:
        return "{0:.4g}M".format(_int / 1000000)
    else:
        return "{0:.4g}G".format(_int / 1000000000)


################################################
#            OS & Python Manipulation          #
################################################


def make_dir(_path, _clear=False):
    """
    Ensures that a given path exists, and if not, attempts to create it.

    :param _path: The Full Directory to create. Note that this will silently ignore None Paths
    :param _clear: If True, will clear any contents previously in the directory if it existed.
    :return: None
    :raises OSError: if unable to create the directory
    """
    if _path is None:
        return

    # First, if asked to clear, then remove directory
    if _clear and os.path.isdir(_path):
        shutil.rmtree(_path)

    # Now (Re)Create
    try:
        os.makedirs(_path)
    except OSError:
        if not os.path.isdir(_path):
            raise


def remove(_path):
    """
    Deletes a file (path) if exists, and ignores if it does not

    :param _path: path to delete
    :return: None
    """
    with contextlib.suppress(FileNotFoundError):
        os.remove(_path)