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

from scipy.optimize import linear_sum_assignment as hungarian
from scipy.special import gamma
from scipy.stats import entropy
from functools import reduce
import numpy as np
import itertools


################################################################
#                       Array Operations                       #
################################################################


def masked_logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """
    Compute the log of the sum of exponentials of input elements. This is identical to the scipy.special function,
       but ignores Masked (NaN) values.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.
    """
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.0  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.nanmax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        s = np.nansum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def exp10(a):
    """
    Raise 10 to array, elementwise

    The function is a convenience similar to numpy's exp and exp2

    :param a:   Arraylike or scalar to raise 10 to.
    :return:    Numpy Array (or scalar) of same shape and type as array, with values 10 ^ array
    """
    return np.power(10.0, a)


def clamp(a, cutoff, limits):
    """
    Clamp the values in a to limits at the cutoff: specifically, values less than or equal to the
    cutoff are converted to limits[0] and values greater than cutoff are converted to limits[1].
    This is a counterpart to the clip function.

    :param a:       Array
    :param cutoff:  Cutoff value for clamping from
    :param limits:  Limits to clamp values to (min/max)
    :return:        Clamped matrix
    """
    aa = a.copy()
    aa_cutoff = aa <= cutoff
    aa[aa_cutoff] = limits[0]
    aa[~aa_cutoff] = limits[1]
    return aa


def mad(a, b, axis=None, keepdims=False):
    """
    Compute the Mean Absolute Deviation between two arrays: they must be of same size or
    broadcastable

    :param a:           First Array
    :param b:           Second Array
    :param axis:        Axes along which to compute MAD
    :param keepdims:    Whether to keep dimensions
    :return:            MAD Array
    """
    return np.mean(np.abs(a - b), axis=axis, keepdims=keepdims)


def multiply(arrays):
    """
    Multiply (element-wise) a series of arrays: they must all be of the same shape!

    :param arrays:  Tuple/list of arrays to multiply
    :return:        Element-wise multiple
    """
    return reduce(np.multiply, arrays)


def value_map(array, _to, _from=None, shuffle=False):
    """
    Map Values in a Numpy Array from a domain to another.

    Currently supports only 1D arrays

    :param array:  Array to map
    :param _to:    Values to map to
    :param _from:  Values to map from: if not present, it is assumed that the entries in to are
                   ordered in their domain [0, len(to)-1]
    :param shuffle: Optimisation: if True, does not need to use searchsorted (uses just array
                    indexing). Only set this to True iff _from contains all contiguous values
                    from 0 up to len(_to) - 1 (also no duplicates).
    :return:       Mapped Array
    """
    # If just shuffling, then use just array indexing...
    if shuffle:
        if _from is not None:
            sort_idx = np.argsort(_from)
            return np.asarray(_to)[sort_idx][array]
        else:
            return np.asarray(_to)[array]

    # Otherwise, have to use search sorted: in this case, from cannot be None!
    else:
        sort_idx = np.argsort(_from)
        idx = np.searchsorted(_from, array, sorter=sort_idx)
        return np.asarray(_to)[sort_idx][idx]


def run_lengths(a, how="I", return_values=False, return_positions=False):
    """
    Compute the length of continuous runs of the same values in an array.

    :param a: Array If not one-d, the input is flattened.
    :param how: How to treat values including np.NaN
                    A: use All values (including treating NaN as its own de-facto value)
                    I: Ignore NaN values but treat all others as before. Note that if an NaN
                       interrupts a stream of same values, then these will be treated as SEPARATE
                       streams.
                    O: compute Only NaNs
    :param return_values: If true, return also the Key (value) for each run.
    :param return_positions: If true, return also the start position of each run.
    :return: lengths, [values], [positions]
               The first array contains the run lengths: optionally, the next array (if present) is
               the values for each run, and finally (if present) the position of the start of
               each run.
    """
    # Prepare
    #  We fist compute assuming that we are going to use everything! This allows us to have correct
    #  position information. We also need to convert arrays of NaN to lists so that NaN works.
    #  This will slow down but cannot be helped.
    if type(a) == np.ndarray:
        a = [i if not np.isnan(i) else np.nan for i in a.flatten()]
    rls = np.asarray(
        [(sum(1 for _ in l), n) for n, l in itertools.groupby(a)]
    )  # Now, compute run-lengths and types.
    pos = (
        np.asarray([0, *np.cumsum(rls[:, 0])[:-1]], dtype=int) if len(rls) > 0 else []
    )  # Compute positions just in case.
    # Compute: Branch on Format
    if how.lower() == "a":
        lens, vals = (rls[:, 0], rls[:, 1]) if len(rls) > 0 else ([], [])
    elif how.lower() == "i":
        n_nan = ~np.isnan(rls[:, 1])
        lens, vals, pos = (
            (rls[n_nan, 0], rls[n_nan, 1], pos[n_nan]) if len(n_nan) > 0 else ([], [], [])
        )
    elif how.lower() == "o":
        nan = np.isnan(rls[:, 1])
        lens, vals, pos = (rls[nan, 0], rls[nan, 1], pos[nan]) if len(nan) > 0 else ([], [], [])
    else:
        raise ValueError('Incorrect format for "how" parameter.')
    # Now Return
    to_return = [lens]
    if return_values:
        to_return.append(vals)
    if return_positions:
        to_return.append(pos.astype(int))
    return tuple(to_return) if len(to_return) > 1 else to_return[0]


def array_nan_equal(left, right):
    """
    Compares two numpy arrays to test for equality, but considers np.nan values to be equal to each
    other
    :param left: The Left array to consider
    :param right: The Right array to consider
    :return: True if left is the same shape as right, and all elements are equal (including nan's at
             same positions)
    """
    # Ensure that Arrays (potentially convert from list)
    try:
        left, right = np.asarray(left), np.asarray(right)
    except ValueError:
        return False

    # Check that Shape is correct
    if left.shape != right.shape:
        return False

    # Return
    return bool(
        np.logical_or(
            np.asarray(left == right), np.logical_and(np.isnan(left), np.isnan(right))
        ).all()
    )


def round_to_multiple(x, base, how="r"):
    """
    Round x's elements to the nearest multiple of 'base'

    :param x: Array to round
    :param base: The base (integer/float) to round to multiples of:
    :param how: The method for rounding: 'r' (round), 'f' (floor) or 'c' (ceiling)
    :return: value rounded up to the nearest multiple of base. The data type will always be float
    """
    if how.lower() == "r":
        return np.round(np.divide(x, float(base))) * base
    elif how.lower() == "f":
        return np.floor(np.divide(x, float(base))) * base
    elif how.lower() == "c":
        return np.ceil(np.divide(x, float(base))) * base
    else:
        raise ValueError("HOW must be one of r/f/c")


################################################################
#                     Matrix Manipulations                     #
################################################################


def non_diag(a: np.ndarray):
    """
    Remove the Diagonal Entries from a matrix

    :param a:   2D Numpy Array to operate on
    :return:    A copy of the matrix with diagonal elements zeroed out
    """
    return a - np.diagflat(a.diagonal())


def make_diagonal(on_diag, off_diag, size):
    """
    Create a Diagonally-dominant matrix with on_diag elements on the main diagonal, and off_diag
    elsewhere

    :param on_diag:     Scalar: Value to put on diagonal
    :param off_diag:    Scalar: Value to put off the diagonal
    :param size:        Scalar: Size of the matrix
    :return:            Diagonal matrix
    """
    return np.eye(size) * on_diag + non_diag(np.full(shape=[size, size], fill_value=off_diag))


def maximise_trace(x):
    """
    Maximise the Trace of a SQUARE Matrix X using the Hungarian Algorithm

    :param x: Numpy 2D SQUARE Array
    :return: Tuple containing (in order):
                * optimal permutation of columns to achieve a maximal trace
                * size of this trace
    """
    _rows, _cols = hungarian(np.full(len(x), np.max(x)) - x)
    return _cols, x[_rows, _cols].sum()


def swap_columns(x, cols):
    """
    Swap the columns of a 2D Matrix

    :param x:    Numpy 2D Array
    :param cols: A 2-tuple, of the column indices to swap
    :return:     Matrix with cols swapped. A copy is always returned.
    """
    temp = x.copy()
    temp[:, cols[0]] = x[:, cols[1]]
    temp[:, cols[1]] = x[:, cols[0]]
    return temp


def ensure2d(a, axis=0):
    """
    Returns a matrix of dimensionality 2 from a potentially linear vector

    :param a: Numpy array
    :param axis: Axis to append if missing: either 0 or 1
    :return: 2D Matrix
    """
    if np.ndim(a) > 1:
        return a
    elif np.ndim(a) == 1:
        return a[np.newaxis, :] if axis == 0 else a[:, np.newaxis]
    else:
        return a * np.ones([1, 1])


################################################################
#                    Probabilistic Helpers                     #
################################################################


class Dirichlet:
    """
    An alternative Dirichlet Class which
        a) Precomputes the normalisation parameters for efficiency
        b) can vectorise over matrices, where the last-dimension is the probability distribution.
        c) Handles 0's in the x's
    """

    def __init__(self, alpha, ignore_zeros=False):
        """
        Initialise the Dirichlet with a particular alpha

        :param alpha: Alpha (Concentration) parameter. This is a K-D Matrix, where the last
                      dimension is the probability
        :param ignore_zeros: If true, will ignore dimensions which are expected to be 0. This is
                             inferred from the alpha matrix, specifically ignoring dimensions
                             which are 1. This is based on the assumption that such dimensions
                             will always be zero and will be known in advance.
        """
        alpha = np.asarray(alpha)  # K-Dimensional
        # Do some Checks
        if np.any(alpha <= 0):
            raise ValueError("All dimensions of Alpha must be greater than 0.")
        self._alpha_m1 = alpha - 1.0  # K Dimensional
        self.__zeros = alpha == 1     # Find location of zeros
        if ignore_zeros:
            # Check that not all masked:
            if self.__zeros.all():
                raise ValueError('If ignoring zeros, you cannot have all of Alpha = 1!')
            # We need to mask both alpha and alpha-1:
            alpha = np.ma.array(alpha, mask=self.__zeros)
            self._alpha_m1 = np.ma.array(self._alpha_m1, mask=self.__zeros)
        self.norm = np.prod(gamma(alpha), axis=-1) / gamma(
            np.sum(alpha, axis=-1)
        )  # K-1 Dimensional
        self.lognorm = np.log(self.norm)  # K-1 Dimensional

    def pdf(self, x):
        """
        Compute the PDF of the passed Matrix (must be of the same dimensionality and size as Alpha).
        Note that if 0's are passed, then these must align with the alpha's ones.

        :param x: Probability Matrix
        :return: Dirichlet Probability on x
        """
        # Ensure that it is non-zero where zeros are not allowed
        if np.any(x[~self.__zeros] <= 0):
            raise ValueError("0-valued probabilities are only allowed where alpha=1")

        # Note that we can do the below because:
        #   a) If we are not ignoring zeros, then anything raised to 0 is 1 in any case, and since
        #      we are taking product this does not matter
        #   b) If we are ignoring zeros, this is masked by definition, which means that the result
        #      of the power is masked and so is the product!
        # However, just in case in the future we decide to support all ones when ignoring,
        # then we check for all zeros:
        if self.__zeros.all():
            return 1 / self.norm
        else:
            return np.prod(np.power(x, self._alpha_m1), axis=-1) / self.norm

    def logpdf(self, x):
        """
        Compute the Log-PDF of the passed Matrix (must be of the same dimensionality and size as
        Alpha)

        :param x: Probability Matrix
        :return:  Log-Probability on X
        """
        # Ensure that it is non-zero where zeros are not allowed
        if np.any(x[~self.__zeros] <= 0):
            raise ValueError("0-valued probabilities are only allowed where alpha=1")
        # In this case, we must mask to avoid the issues with log since -inf * mask = NaN and not 0
        #   as we would expect. Since we are enforcing that 0-valued probabilities are only
        #   allowed when alpha=1, then we can mask out all such values, because in any case,
        #   _alpha_m1=0 (for all self.__zeros) and hence, the multiplication will be 0 in any case.
        # We do have to handle the case where all of _alpha_m1 is 0, in which case,
        #    the sum should be just 0
        if self.__zeros.all():
            return -self.lognorm
        else:
            x = np.ma.array(x, mask=self.__zeros)
            return np.sum(np.multiply(np.log(x), self._alpha_m1), axis=-1) - self.lognorm

    def logsumpdf(self, x):
        """
        Compute the Log-PDF of the passed Matrix (must be of the same dimensionality and size as
        Alpha) and sum it.

        :param x: Probability Matrix
        :return:  Log-Sum-Probability on X
        """
        return self.logpdf(x).sum()

    def sample(self):
        """
        Generates a single sample from the Dirichlet

        :return:    A single sample
        """
        return self._recursive_sample(self._alpha_m1 + 1)

    @staticmethod
    def _recursive_sample(alpha):
        # Handle Base-Case with just 1 dimension
        if alpha.ndim == 1:
            return np.random.dirichlet(alpha)
        # Handle Other Cases
        else:
            sample = np.empty_like(alpha)
            for i in range(alpha.shape[0]):
                sample[i, :] = Dirichlet._recursive_sample(alpha[i, :])
            return sample


def sum_to_one(x, axis=None, norm=False):
    """
    Ensure that the elements of x sum to 1 (normally for probabilities), by dividing by their sum.

    The function avoids division by-zero errors.

    :param x:       Array to normalise
    :param axis:    If not None (default) specifies axis to normalise across: otherwise,
                    normalisation happens across flattened array.
    :param norm:    If Norm, return also the normaliser
    :return: Normalised Array
    """
    _sum = np.sum(x, axis=axis, keepdims=True)  # Find Sum
    _sum[_sum == 0] = 1.0  # Avoid Division by Zero
    return (np.divide(x, _sum), 1.0 / _sum.squeeze()) if norm else np.divide(x, _sum)


def invert_softmax(x, enforce_unique=None):
    """
    Return a vector which would yield x under softmax: note that uniqueness is achieved in a number
    of ways.

    :param x: Vector/Matrix of probabilities (last dimension must sum to 1)
    :param enforce_unique: If None (default) then uniqueness is achieved by enforcing that the
                            inverted space sums to 0. Otherwise, it specifies an index which is
                            enforced to be 0.
    :return:  Inverse Softmax
    """
    # Compute Log(X)
    log_x = np.log(x)

    # Branch on how we achieve uniqueness
    if enforce_unique is None:
        return log_x - np.mean(
            log_x, axis=-1, keepdims=True
        )  # Eq. (7), substituting Eq. (8) for C.
    else:
        return log_x - np.expand_dims(
            log_x[..., enforce_unique], -1
        )  # Eq. (7), substituting Eq. (9) for C.


def conditional_entropy(emission, prior=None, base=None):
    """
    Compute the Conditional Entropy of a Joint Distribution over latent and conditioned (visible)
    variables. This only supports 2D Emissions (ie one latent variable and one visible)

    :param emission:    The L by V conditional probabilities (Latent along the first index)
    :param prior:       The Distribution over L (if None, use vector of equi-probability)
    :param base:        The numeric base to operate with.
    :return:            Conditional Entropy (scalar)
    """
    if emission.ndim == 2:
        if prior is None:
            prior = np.ones(len(emission)) / len(emission)
        return np.matmul(
            prior, entropy(emission.T, base=base)
        )  # Sum_{l=1}^{|L|} P(L=l) H(V|L=l): see NIP slide 6

    else:
        raise ValueError(
            "Function does not support Tensors of dimensionality {}. If you need to compute"
            " for a latent variable of dimensionality 2, then use conditional_entropy_2D()".format(
                emission.ndim
            )
        )


def conditional_entropy_2D(emission, prior=None, base=None):
    """
    Compute the Conditional Entropy of a joint distribution over latent states and conditioned
    variables This version is designed to be used when the conditioning variables (latent) are
    2D: NOT when the visible ones are 2D

    :param emission: The L1 by L2 by V conditional probabilities (3D with latents along first two
                     indices)
    :param prior:    The distribution over L1/L2: if None, will initialise to uniform probability
    :param base:     The numeric base to operate with.
    :return:         Conditional Entropy (scalar)
    """
    if emission.ndim == 3:
        # Get the Latent dimensionalities
        s1, s2, _ = np.shape(emission)
        # Construct Prior if not provided
        if prior is None:
            prior = sum_to_one(np.ones([s1, s2]))
        # Sum over combinations of each L1/L2
        cond_ent = 0
        for r in range(s1):
            for c in range(s2):
                cond_ent += prior[r, c] * entropy(emission[r, c, :], base=base)
        return cond_ent

    else:
        raise ValueError(
            "Function does not support Tensors of dimensionality {}. For standard conditional "
            " entropy, use conditional_entropy().".format(emission.ndim)
        )


def mutual_information(prior, emission, base=None):
    """
    Compute the Mutual information between an input (Z) and set of output (X) variables, under the
      assumption that when there is more than 1 X variable, they are conditionally independent of
      each other given Z (i.e. the Naive Bayes assumption). Be careful however, that as the
      number of X variables increases, the dimensionality of the problem explodes (since we need
      to work with the full joint marginal)!

    Note (1) - This is NOT the Conditional Mutual Information, which is something different: this is
      just the MI between a Z and a multi-dimensional X (taken as one variable).

    Note (2) - This only supports a 1D latent variable (Z): i.e. there may be multiple (X) variables
      but only 1 (Z).

    :param prior:       Prior Distribution over Z [1D array]
    :param emission:    Conditional Distribution over X given Z. This can be either:
                            a) 2D Numpy array, with Z along the rows, for one conditional
                            b) List of 2D Numpy arrays, each constituting a 2D Numpy array (Z along
                               rows) showing the emission of a variable.
    :param base:        The numeric base to operate with.
    :return:            Mutual Information
    """
    # First Collapse all emissions into 1 by computing outer product along X-axis (i.e. finding the
    #  cross-product) This is done by a smart use of the self-looping and the outer product
    #  function. Basically, we need to find the cross-combination of all emissions. To do this,
    #  for each latent state (Z=z), we find the probability of each combination of Xs by
    #  repeatedly (over each variable) finding the outer product between the current flattened
    #  representation and the next variable in sequence.
    if type(emission) in (list, tuple):
        pXZ = [1 for _ in prior]
        for k in range(len(prior)):
            for variable in emission:
                pXZ[k] = np.outer(
                    pXZ[k], variable[k, :]
                ).ravel()  # We are continuously increasing in size...
        emission = np.asarray(pXZ)  # Will be array of size |Z| by |X_1|*|X_2|*...|X_n|

    # Compute Marginal X:
    pX = np.matmul(prior, emission)

    # Now Compute Entropies and return
    return entropy(pX, base=base) - conditional_entropy(emission=emission, prior=prior, base=base)


def markov_stationary(transition):
    """
    Return the Stationary Distribution of a Markov-Chain Transition Matrix

    :param transition: The MC Transition probabilities
    :return: The Stationary Distribution
    """
    evals, evect = np.linalg.eig(transition.T)
    _i = np.where(np.isclose(evals, 1.0))[0][0]

    return sum_to_one(np.real_if_close(evect[:, _i]))


def switch_rate(a, axis=-1, ratio=False):
    """
    Compute the number/frequency of switching at various levels in a time-series.
    This is guaranteed to work for 1D/2D arrays but the result is unspecified for other
    dimensionalities

    :param a:       Array_like
    :param axis:    Optional: the axis along which to compute, default is the last axis. Note that
                    this can only be one axis at most.
    :param ratio:   Whether to return a ratio or absolute value.
    :return:
    """
    # Get the Difference
    a_diff = (np.diff(a, n=1, axis=axis) != 0).sum(axis=axis)

    # Now get the shape if need be
    a_len = np.shape(a)[axis]

    # Return
    return a_diff / a_len if ratio else a_diff
