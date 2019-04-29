"""
Other General Utilities
"""
import numpy as np
import shutil
import copy
import os


################################################
#            Collections Processing            #
################################################

def to_tuple(value):
    """
    Static (Module) Method for converting a scalar into a tuple if not already (tuple/list/numpy array). Will also
    ignore None. Note, that this does not handle dict types!

    :param value: The Value to convert
    :return: A Tuple containing value if a scalar, or value if already a list/tuple/numpy array/none
    """
    return value if (type(value) in (tuple, list, np.ndarray) or value is None) else (value,)


def to_list(value):
    """
    Static (Module) Method for converting a scalar into a list if is not already an iterable (tuple/list/array)

    :param value: The Value to convert
    :return: A List containing value if a scalar, or value if already a list/tuple/numpy array/none
    """
    return value if (type(value) in (tuple, list, np.ndarray) or value is None) else [value, ]


def to_scalar(value):
    """
    Converts a single-element tuple into a scalar value if not already. To do this, it attempts to dereference the
    first element if it is of type tuple, list or np.ndarray

    :param value:
    :return:
    """
    return value[0] if (type(value) in (tuple, list, np.ndarray)) else value


def extend_dict(_d1, _d2, deep=False):
    """
    This function may be used to extend the 'list/array'-type elements of _d1 by corresponding entries in _d2. The elements in either case must support the extend
    method (i.e. are typically lists) - scalars are however supported through the lister method. Note that if a key
    exists in _d2 and not in _d1, it is automatically created as a list.

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
        yield (i, tuple(d[i] for d in _dcts))


def dict_invert(_dict):
    """
    Inverts the Key-Value pairs in a dictionary.
    ***IMP***: They Value entries themselves must be unique
    The function also works on lists (putting values as keys and the index in the list as values

    :param _dict: The dictionary to invert or a list to extract from
    :return:      A new dict object which the key-value entries reversed (i.e. values become keys and v.v.)
    """
    if type(_dict) is dict:
        return {v: k for k, v in _dict.items()}
    else:
        return {v: k for k, v in enumerate(_dict)}


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
        self.write('\n')
        self.flush()


def name(obj):
    """
    Static (Module) Method for outputting the name of a data type. This amounts to calling the __name__ method on
    the passed obj (or its class type) if it exists, or calling string otherwise on it... (mainly for None Types)

    :param obj: An instance or class type.
    :return: string representation of 'obj'
    """
    if isinstance(obj, type):
        if hasattr(obj, '__name__'):
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
    return ['{1:.{0}f}'.format(prec, f) for f in _list]


def str_width(_iterable):
    """
    Returns the maximum width of all strings in iterable

    :param _iterable:
    :return:
    """
    return max([len(str(s)) for s in _iterable])


def dict_width(_dict):
    """
    Returns the maximum length of all elements in a dictionary. Note that if the dict contains strings, strwidth should
    be used, as this will always return 1.

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
        return '{0:.4g}K'.format(_int/1000)
    elif _int < 1000000000:
        return '{0:.4g}M'.format(_int/1000000)
    else:
        return '{0:.4g}G'.format(_int/1000000000)


################################################
#                OS Manipulation               #
################################################

def make_dir(_path, _clear=False):
    """
    Static (Module) Method for ensuring that the given path exists, and if not, will attempt to create it.

    :param _path: The Full Directory to create
    :param _clear: If True, will clear any contents previously in the directory if it existed.
    :return: None
    :raises OSError: if unable to create the directory
    """
    # First, if asked to clear, then remove directory
    if _clear and os.path.isdir(_path):
        shutil.rmtree(_path)

    # Now (Re)Create
    try:
        os.makedirs(_path)
    except OSError:
        if not os.path.isdir(_path): raise
