"""
Other General Utilities
"""
import numpy as np
import copy

################################################
##           Collections Processing           ##
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





class NullableSink:
    """
    Defines a wrapper class which supports a nullable initialisation
    """
    def __init__(self, obj=None):
        self.Obj = obj

    def write(self, str):
        if self.Obj is not None:
            self.Obj.write(str)

    def flush(self):
        if self.Obj is not None:
            self.Obj.flush()
