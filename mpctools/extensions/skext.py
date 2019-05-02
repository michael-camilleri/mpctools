"""
This module contains some extensions to Scikit-Learn (hence the name 'SKlearn EXTensions').

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
http://www.gnu.org/licenses/.
"""
from mpctools.extensions import npext
import numpy as np


def hierarchical_log_loss(y_true, y_prob, mapping, eps=1e-15):
    """
    Compute the Log-Loss, when y_true contains over-arching labels which are not predictable in y_prob.

    :param y_true:
    :param y_prob:
    :param mapping: For each super-label, the set of label probabilities which must be summed. Must be a dictionary of
                    arrays, since will not start from 0. It is assumed that the super-labels are contiguous and follow
                    the fine-grained labels immediately, which are themselves numbered zero through L-1
    :return:
    """
    # Get some sizes first:
    _l, _d = y_prob.shape

    # Now allocate log-loss array
    _ll = np.empty(_l)

    # First deal with those which are in the fine-grained set
    _ll[y_true < _d] = y_prob[y_true < _d, y_true[y_true < _d].astype(int)]

    # Now do those which are in the mapping
    for m_s, m_f in mapping.items():
        _ll[y_true == m_s] = np.multiply(y_prob[y_true == m_s, :], np.asarray(m_f)[np.newaxis, :]).sum(axis=1)

    # Finally compute the actual log-loss (after clipping)
    return -np.sum(np.log(np.clip(_ll, eps, 1)))/_l


def multi_way_split(y, sizes, splitter, random_state=None):
    """
    Convenience Function for wrapping a multi-way split. This only returns the indices of the split.
    This is actually implemented as a recursive function.

    :param y:               The target labels. If using a stratified splitter, then this must be the true targets:
                            otherwise, it is enough to be an empty array of the same length as the data.
    :param sizes:           The relative sizes of the three sets. Note that these should sum to 1.
    :param splitter:        The splitting object: this allows stratified/unstratified type splits (basically one of
                            ShuffleSplit or StratifiedShuffleSplit)
    :param random_state:    Any random state to employ
    :return:                N-tuple of indices, one each for the N sets.
    """
    # --- In either case, ensure that the sizes sum to 1! --- #
    sizes = npext.sum_to_one(sizes)
    # --- Base Case: We know how to do this --- #
    if len(sizes) == 2:
        return next(splitter(n_splits=1, train_size=sizes[0], test_size=sizes[1],
                             random_state=random_state).split(y, y))
    # --- Other Cases --- #
    #   This is a bit trickier. We have to first split assuming that all but the first set are grouped together. We then
    #   pass the second set of targets recursively to our function, with the remaining sizes. However, when the indices
    #   are returned, they must be remapped to the original index set, since they are indices into that set.
    else:
        sub_sizes = sizes[1:]
        left, right = next(splitter(n_splits=1, train_size=sizes[0], test_size=np.sum(sub_sizes),
                                    random_state=random_state).split(y, y))
        right_split = multi_way_split(y[right], sub_sizes, splitter, random_state + 1 if random_state is not None else None)
        idcs = [left]
        for i in right_split:
            idcs.append(right[i])
        return idcs