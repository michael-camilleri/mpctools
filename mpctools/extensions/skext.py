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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.calibration import CalibrationDisplay
from sklearn import preprocessing as skpreproc
from scipy.spatial.distance import squareform
from mpctools.extensions import npext, utils
from sklearn import metrics as skmetrics
from numba import njit, float64, types
from scipy import stats as scstats
from scipy.special import softmax
from sklearn.svm import SVC
import torch.nn as tnn
import numpy as np
import warnings
import joblib
import torch
import copy


class ThresholdedClassifier:
    """
    Simple wrapper which applies a threshold to a classifier.

    Currently, only works for Binary Classifiers
    """
    def __init__(self, clf, threshold=0.5):
        self.__clf = clf
        self.__thr = threshold

    def fit(self, X, y):
        """
        Calls the underlying classifier's fit method (does not fit the threshold)
        """
        return self.__clf.fit(X, y)

    def predict(self, X):
        """
        Returns positive class if probability is above threshold, else negative class.
        """
        return (self.__clf.predict_proba(X)[:, 1] > self.__thr).astype(int)

    def predict_proba(self, X):
        """
        Calls the underlying predict_proba() method for the classifier.
        """
        return self.__clf.predict_proba(X)

    @property
    def threshold(self):
        return self.__thr


class SVCProb:
    """
    Wrapper around the SVC Classifier to include a probability interpretation of the DF (as
    opposed to the CV-folds based estimation).
    This is less accurate but provides a rough estimate for ROC thresholds.
    """
    def __init__(
            self,
            C=1.0,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=0.001,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr',
            break_ties=False,
            random_state=None
    ):
        """
        Calls the underlying Initialiser and also the Min-Max Scaler.
        """
        self.__svc = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=False, # The only forced parameter
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state
        )
        self.__dfs = skpreproc.MinMaxScaler(clip=True) if probability else None

    def get_params(self, deep=True):
        """
        Wrapper around the parameters of the SVC estimator.
        """
        params = self.__svc.get_params(deep)
        params['probability'] = self.__dfs is not None
        return params

    def fit(self, X, y):
        """
        Fits both the SVC and the scaler for the DFS. (Mostly to conform to sklearn framework)
        """
        self.__svc.fit(X, y)
        if self.__dfs is not None:
            self.__dfs.fit(self.__svc.decision_function(X).reshape(-1, 1))
        return self

    def predict(self, X):
        return self.__svc.predict(X)

    def predict_proba(self, X):
        if self.__dfs is not None:
            probs = self.__dfs.transform(self.__svc.decision_function(X).reshape(-1, 1))
            return np.hstack([1-probs, probs])
        else:
            raise RuntimeError('No Probability Was Setup.')


class MixtureOfCategoricals:
    """
    Implements a multi-variate mixture of categoricals similar to the Dawid-Skene Model but with
      potentially varying latent and emission dimensionality. Note however that the dimensionality
      should be the same for all variables.

    Note:
        Indexing for Psi is always K, Z, X
        The class can handle missing data (including during learning), represented by NaN
    """

    def __init__(
        self, sZ, sKX, alpha_pi=None, alpha_psi=None, init_pi=None, init_psi=None, tol=1e-4,
        inits=1, random_state=None, max_iter=100, n_jobs=-1
    ):
        """
        Initialises the Class

        :param sZ: Size of the Latent Dimension |Z| (or) alternatively, an Array initialiser for Pi.
        :param sKX: Number of Variables and Size (K, X) (or) alternatively, an Array initialiser
                for Psi.
        :param alpha_pi: Array-Like of size Z dictating the alpha-prior over Pi: if None, defaults
                to uninformative prior alpha=1
        :param alpha_psi: Alpha parameters for prior over Psi i.e. indexing is [k][z][x]. If None,
                defaults to uninformative alpha=1 priors
        :param init_pi: Array-Like of size Z governing sampling Dirichlet for random restarts: if
                None, uses the same as the prior
        :param init_psi: List of lists for sampling Psi: if None, uses the prior.
        :param tol: Tolerance for convergence (stopping criterion)
        :param inits: Number of random restarts. These are random initialisations from the provided
                priors.
        :param random_state: Random Generator State (used to initialise the probability parameters)
        :param max_iter: Maximum number of iterations taken for the solvers to converge.
        :param n_jobs: Number of cores used when parallelising over runs. If > 0 uses
                multiprocessing, if < 0 uses threading. 0 indicates serial execution.
        """
        # Resolve Sizes/Parameters
        if isinstance(sZ, np.ndarray):
            self.__pi = np.array(sZ, dtype=float, copy=True)
            self.sZ = len(sZ)
        else:
            self.__pi = None
            self.sZ = sZ
        if isinstance(sKX, np.ndarray):
            self.__psi = np.array(sKX, dtype=float, copy=True)
            self.sK = sKX.shape[0]
            self.sX = sKX.shape[-1]
        else:
            self.__psi = None
            self.sK = sKX[0]
            self.sX = sKX[1]
        # Copy other parameters
        self.__inits = inits
        self.__tol = tol
        self.__max_iter = max_iter
        self.__n_jobs = n_jobs
        # Resolve Priors
        self.__pi_prior = utils.default(alpha_pi, np.ones(self.sZ))
        self.__psi_prior = utils.default(
            alpha_psi, [[np.ones(self.sX) for _ in range(self.sZ)] for _ in range(self.sK)]
        )
        self.__pi_init = utils.default(init_pi, self.__pi_prior.copy())
        self.__psi_init = utils.default(init_psi, copy.deepcopy(self.__psi_prior))
        # Create Random Generator
        self.__rnd = np.random.default_rng(random_state)
        # Finally, empty set of fit parameters
        self.__fit_params = []
        self.__best = None

    def sample(self, N, as_probs=False, noisy=None):
        """
        Generate Samples from the distribution

        Generates 'n' samples from the distribution, which are returned as two lists.

        :param N: Number of samples to generate
        :param as_probs: If True, returns probabilities (rather than integer samples)
        :param noisy: If as_probs is true, and this is not None, it encodes the noise for
        sampling from:
            * If a scalar, it is an additive constant to the dirichlet such that the X's are
            sampled from a dirichlet with alpha = (noisy, ..., 1+noisy, ..., noisy)
            * Otherwise an ndarray, containing the conditional alphas to sample from.
        :return: Tuple with two entries
            * Z : Latent variable samples N (x Z)
            * X : Emission symbols, N x K (x X)
        """
        # Check that model was fit
        if self.Pi is None or self.Psi is None:
            raise RuntimeError('Model must be fit first before sampling from it.')
        _pi, _psi = self.Pi, self.Psi

        # Sample from Model
        _Z = self.__rnd.choice(self.sZ, size=N, p=_pi) # Sample Z at one go
        _X = np.empty([N, self.sK]) # Sampling of X is conditional
        for n in range(N):
            for k in range(self.sK):
                _X[n, k] = self.__rnd.choice(self.sX, size=1, p=_psi[k, _Z[n], :])

        if as_probs:
            # For Z just fill in ones in the ordinal location.
            _ZProb = np.zeros([N, self.sZ])
            for z in range(self.sZ):
                _ZProb[_Z == z, z] = 1
            _Z = _ZProb
            # For X, will need to iterate over cases
            _XProb = np.zeros([N, self.sK, self.sX])
            for x in range(self.sX):
                _mask = _X == x
                if noisy is not None:
                    if isinstance(noisy, np.ndarray):
                        _alpha = noisy[x, :]
                    else:
                        _alpha = np.ones(self.sX) * noisy; _alpha[x] += 1
                    _XProb[_mask, :] = scstats.dirichlet.rvs(_alpha, _mask.sum(), self.__rnd)
                else:
                    _XProb[_X == x, x] = 1
            _X = _XProb

        # Return
        return _Z, _X

    def fit(self, X, z=None):
        """
        Fits the Model

        Fits the Pi/Psi parameters using EM.

        :param X: The Observations to fit on: N x K x X. Note that along the last dimension,
        the vector may be all NaNs to indicate missing observation for k @ sample n.
        :param z: None: used for compatibility with sklearn framework
        :return: self, for chaining.
        """
        # Create Starting Points
        #   - These are sampled from the priors
        start_pi = np.asarray([
            np.array(scstats.dirichlet.rvs(self.__pi_init, 1, self.__rnd).squeeze(), ndmin=1)
            for _ in range(self.__inits)
        ])
        start_psi = np.asarray([
            [
                [scstats.dirichlet.rvs(alpha_kz, 1, self.__rnd).squeeze() for alpha_kz in alpha_k]
                for alpha_k in self.__psi_init
            ]
            for _ in range(self.__inits)
        ])

        # Run Multiple runs (in parallel):
        if self.__n_jobs == 0:  # Run serially
            self.__fit_params = [
                self.partial_fit(X, starts) for starts in zip(start_pi, start_psi)
            ]
        else:  # Run in parallel using multiprocessing or threads
            mode = 'threads' if self.__n_jobs < 0 else 'processes'
            self.__fit_params = joblib.Parallel(n_jobs=abs(self.__n_jobs), prefer=mode)(
                joblib.delayed(self.partial_fit)(X, starts) for starts in zip(start_pi, start_psi)
            )

        # Select best
        # Check if any converged
        if np.all([not fp[3] for fp in self.__fit_params]):
            warnings.warn('None of the runs converged.')
        # Find run with maximum ll
        self.__best = np.argmax([fp[2][-1] for fp in self.__fit_params])
        self.__pi = self.__fit_params[self.__best][0].copy()
        self.__psi = self.__fit_params[self.__best][1].copy()

        # Return Self
        return self

    def predict_proba(self, X):
        """
        Predicts probability over latent-state Z

        :param X: X-values (N x K x X)
        :return: probabilities over Z (N x Z)
        """
        return self.__responsibility(X, self.Pi, self.Psi)[0]

    def predict(self, X):
        """
        Argmax of predict proba
        :param X:
        :return:
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def partial_fit(self, X, p_init=None):
        """
        Private method for fitting a single initialisation of the parameters using EM

        :param X: The Observations to fit on: see fit above
        :param p_init: Initial Value for the Pi/Psi probabilities: if not specified, uses the
                current member variables.
        :param optim: Optimisation parameters: max_iter, tol, pi_alpha, psi_alpha
        :return: Tuple with:
            * Pi: Fit Pi
            * Psi: Fit Psi
            * Log-Likelihoods: Log-Likelihood evolution
            * Converged: True if converged, False otherwise
        """
        # Resolve Parameters
        if p_init is None:
            if self.Pi is None or self.Psi is None:
                raise ValueError('Cannot initialise from self if no initial values for Pi/Psi')
            pi, psi = self.Pi, self.Psi
        else:
            pi, psi = p_init
        dir_pi = scstats.dirichlet(self.__pi_prior)
        dir_psi = [[scstats.dirichlet(a_kz) for a_kz in a_k] for a_k in self.__psi_prior]

        # Prepare to Run
        iters = 0
        log_likelihood = []

        # Run EM
        while iters < self.__max_iter:
            # ---- E-Step ---- #
            # 1) Compute Responsibilities
            gamma, ll = self.__responsibility(X, pi, psi)
            # 2) Update Log-Likelihood, including contribution of Pi/Psi
            ll += dir_pi.logpdf(pi)
            for k in range(self.sK):
                for z in range(self.sZ):
                    ll += dir_psi[k][z].logpdf(psi[k, z, :])
            log_likelihood.append(ll)

            # ---- Check for Convergence ---- #
            if self.__converged(log_likelihood):
                break

            # ---- M-Step ---- #
            # 1) Update Pi
            pi = npext.sum_to_one(gamma.sum(axis=0) + self.__pi_prior - 1)
            # 2) Update Psi (per dimension)
            psi = np.asarray([[a_kz - 1 for a_kz in a_k] for a_k in self.__psi_prior])
            self.__update_psi(X, gamma, psi)
            psi = npext.sum_to_one(psi, axis=-1)

            # ---- Iterations ---- #
            iters += 1

        # Return result
        return pi, psi, log_likelihood, self.__converged(log_likelihood)

    def logpdf(self, X):
        """
        Return the (evidence) log-likelihood for the data

        :param X: The observations to compute the evidence log-likelihood for: N x K x X
        :return: Log-Likelihood: note that this does not include the prior likelihood
        """
        return self.__responsibility(X, self.Pi, self.Psi)[1]

    def BIC(self, X):
        return self.__free_params() * np.log(X.shape[0]) - 2 * self.logpdf(X)

    def AIC(self, X):
        return 2 * self.__free_params() - 2 * self.logpdf(X)

    @property
    def Evolution(self):
        return np.asarray(self.__fit_params[self.__best][2])

    @property
    def Stability(self):
        return np.asarray([fp[2][-1] for fp in self.__fit_params])

    @property
    def Converged(self):
        return self.__fit_params[self.__best][3]

    @property
    def Pi(self):
        return self.__pi.copy()

    @property
    def Psi(self):
        return self.__psi.copy()

    def __free_params(self):
        return self.sZ - 1 + self.sK * self.sZ * (self.sX - 1)

    def __converged(self, lls):
        """
        Convergence Check

        :param lls: Array of Log-Likelihood
        :param tol: Tolerance Parameter
        :return: True only if converged, within tolerance
        """
        if len(lls) < 2:
            return False
        elif lls[-1] < lls[-2]:
            warnings.warn("Drop in Log-Likelihood Observed! Results are probably wrong.")
            return False
        else:
            return abs((lls[-1] - lls[-2]) / lls[-2]) < self.__tol

    @staticmethod
    def __responsibility(X, pi, psi):
        """
        Compute the responsibilities (probability over Z)

        :param X: Data: Array-like of size N x |X|
        :param pi: Pi probability
        :param psi: Psi probability
        :return: Probability over Z's, as well as log-likelihood (normaliser).
        """
        # Some sizes
        sN, sK = X.shape[:2]

        # Prepare Gamma and evaluate
        gamma = np.tile(pi[np.newaxis, :], [sN, 1])
        MixtureOfCategoricals.__gamma(X, psi, gamma)
        gamma, ll = npext.sum_to_one(gamma, axis=1, norm=True)

        # Return Gamma and Log-Likelihood
        return gamma, -np.log(ll).sum()

    @staticmethod
    @njit(signature_or_function=(types.Array(float64, 3, 'C', readonly=True), float64[:,:,:], float64[:,:]))
    def __gamma(X, psi, gamma):
        """
        JIT Wrapper for computing Gamma

        :param X: Observations (2D Array). Can handle NaN's
        :param psi: Emission probabilities
        :param gamma: Current version of Gamma
        :return: None (gamma is modified in place)
        """
        sK = X.shape[1]
        sN, sZ = gamma.shape
        for n in range(sN):
            for k in range(sK):
                if np.isfinite(X[n, k, :]).all():  # Only Proceed if Not Missing
                    for z in range(sZ):
                        gamma[n, z] *= np.power(psi[k, z, :], X[n, k, :]).prod()

    @staticmethod
    @njit(signature_or_function=(types.Array(float64, 3, 'C', readonly=True), float64[:, :], float64[:, :, :]))
    def __update_psi(X, gamma, psi):
        """
        Convenience wrapper for updating PSI_k using JIT

        :param X: Observations
        :param gamma: Responsibilities
        :param psi: Psi
        :return: None (Psi is modified in place)
        """
        sK = X.shape[1]
        sN, sZ = gamma.shape
        for n in range(sN):
            for k in range(sK):
                if np.isfinite(X[n, k, :]).all():  # Only Proceed if Not Missing
                    for z in range(sZ):
                        psi[k, z, :] += gamma[n, z] * X[n, k, :]


class CategoricalHMM:
    """
    Implements a Categorical HMM, with support for:
        1. Multiple emissions per Time-Point
        2. Learning from noisy (probability) labels
        3. Learning from missing data
    Some Notes:
        * Indexing for Psi is always K, Z, X
        * Indexing for Omega is always Z^{t-1}, Z^{t}
        * For equation reference MISC_080
    """
    def __init__(
        self, sZ, sKX, omega=None, alpha_pi=None, alpha_psi=None, alpha_omega=None,
            init_pi=None, init_psi=None, init_omega=None, tol=1e-4, inits=1, random_state=None,
            max_iter=100, n_jobs=-1
    ):
        """
        Initialises the Class

        :param sZ: Size of the Latent Dimension |Z| (or) alternatively, an Array initialiser for Pi.
        :param sKX: Number of Variables and Size (K, X) (or) alternatively, an Array initialiser
                for Psi.
        :param omega: If not none, initialise for Omega
        :param alpha_pi: Array-Like of size Z dictating the alpha-prior over Pi: if None, defaults
                to uninformative prior alpha=1
        :param alpha_psi: Alpha parameters for prior over Psi i.e. list of list of arrays with
                indexing [k][z][x]. If None, defaults to uninformative alpha=1 priors
        :param alpha_omega: Alpha parameters for prior over Omega i.e. indexing is [z'][z]. If None,
                defaults to uninformative alpha=1 priors
        :param init_pi: Array-Like of size Z governing sampling Dirichlet for random restarts: if
                None, uses the same as the prior
        :param init_psi: List of lists of arrays for sampling Psi: if None, uses the prior.
        :param init_omega: List of arrays for sampling Omega: if None, uses the prior
        :param tol: Tolerance for convergence (stopping criterion)
        :param inits: Number of random restarts. These are random initialisations from the provided
                priors.
        :param random_state: Random Generator State (used to initialise the probability parameters)
        :param max_iter: Maximum number of iterations taken for the solvers to converge.
        :param n_jobs: Number of cores used when parallelising over runs. If > 0 uses
                multiprocessing, if < 0 uses threading. 0 indicates serial execution.
        """
        # Resolve Sizes/Parameters
        if isinstance(sZ, np.ndarray):
            self.__pi = np.array(sZ, dtype=float, copy=True)
            self.sZ = len(sZ)
        else:
            self.__pi = None
            self.sZ = sZ
        if isinstance(sKX, np.ndarray):
            self.__psi = np.array(sKX, dtype=float, copy=True)
            self.sK = sKX.shape[0]
            self.sX = sKX.shape[-1]
        else:
            self.__psi = None
            self.sK = sKX[0]
            self.sX = sKX[1]
        self.__omega = np.array(omega, dtype=float, copy=True) if omega is not None else None
        # Copy other parameters
        self.__inits = inits
        self.__tol = tol
        self.__max_iter = max_iter
        self.__n_jobs = n_jobs
        # Resolve Priors
        self.__pi_prior = utils.default(alpha_pi, np.ones(self.sZ))
        self.__psi_prior = utils.default(
            alpha_psi, [[np.ones(self.sX) for _ in range(self.sZ)] for _ in range(self.sK)]
        )
        self.__omega_prior = utils.default(
            alpha_omega, [np.ones(self.sZ) for _ in range(self.sZ)]
        )
        self.__pi_init = utils.default(init_pi, self.__pi_prior.copy())
        self.__psi_init = utils.default(init_psi, copy.deepcopy(self.__psi_prior))
        self.__omega_init = utils.default(init_omega, copy.deepcopy(self.__omega_prior))
        # Create Random Generator
        self.__rnd = np.random.default_rng(random_state)
        # Finally, empty set of fit parameters
        self.__fit_params = []
        self.__best = None

    def sample(self, NT, as_probs=False, noisy=None):
        """
        Generate Samples from the distribution

        Generates 'n' samples of lengths 't' from the distribution.

        :param NT: List of sample lengths: will generate len(NT) samples each of length T_n
        :param as_probs: If True, returns probabilities (rather than integer samples)
        :param noisy: If as_probs is true, and this is not None, it encodes the noise for
        sampling from:
            * If a scalar, it is an additive constant to the dirichlet such that the X's are
            sampled from a dirichlet with alpha = (noisy, ..., 1+noisy, ..., noisy)
            * Otherwise an ndarray, containing the conditional alphas to sample from.
        :return: List of two-Tuples with entries:
            * Z : Latent variable samples T_n (x Z)
            * X : Emission symbols, T_n x K (x X)
        """
        # Check that model was fit
        if self.Pi is None or self.Psi is None or self.Omega is None:
            raise RuntimeError('Model must be fit first before sampling from it.')
        _pi, _psi, _omega = self.Pi, self.Psi, self.Omega

        # Sample from Model
        Z, X = [], []
        for t_n in NT:  # Iterate over samples
            # Sample Z
            _Z = np.empty(t_n, dtype=int)
            _Z[0] = self.__rnd.choice(self.sZ, size=1, p=_pi)  # First Entry from Pi
            for t in range(1, t_n):
                _Z[t] = self.__rnd.choice(self.sZ, size=1, p=_omega[_Z[t-1], :])

            # Sample X
            _X = np.empty([t_n, self.sK])  # Sampling of X is also conditional on Z
            for t in range(t_n):
                for k in range(self.sK):
                    _X[t, k] = self.__rnd.choice(self.sX, size=1, p=_psi[k, _Z[t], :])

            # Resolve as Probabilities
            if as_probs:
                # For Z just fill in ones in the ordinal location.
                _ZProb = np.zeros([t_n, self.sZ])
                for z in range(self.sZ):
                    _ZProb[_Z == z, z] = 1
                _Z = _ZProb
                # For X, will need to iterate over cases
                _XProb = np.zeros([t_n, self.sK, self.sX])
                for x in range(self.sX):
                    _mask = _X == x
                    if noisy is not None:
                        if isinstance(noisy, np.ndarray):
                            _alpha = noisy[x, :]
                        else:
                            _alpha = np.ones(self.sX) * noisy; _alpha[x] += 1
                        _XProb[_mask, :] = scstats.dirichlet.rvs(_alpha, _mask.sum(), self.__rnd)
                    else:
                        _XProb[_X == x, x] = 1
                _X = _XProb

            # Append
            Z.append(_Z)
            X.append(_X)

        # Return
        return Z, X

    def fit(self, X, z=None):
        """
        Fits the Model

        Fits the Pi/Psi/Omega parameters using EM.

        :param X: The Observations to fit on: N list of T x K x X. Note that along the last
                dimension, the vector may be all NaNs to indicate missing observation for k @
                sample n, time t.
        :param z: None: used for compatibility with sklearn framework
        :return: self, for chaining.
        """
        # Create Starting Points
        #   - These are sampled from the priors
        start_pi = np.asarray([
            np.array(scstats.dirichlet.rvs(self.__pi_init, 1, self.__rnd).squeeze(), ndmin=1)
            for _ in range(self.__inits)
        ])
        start_psi = np.asarray([
            [
                [scstats.dirichlet.rvs(alpha_kz, 1, self.__rnd).squeeze() for alpha_kz in alpha_k]
                for alpha_k in self.__psi_init
            ]
            for _ in range(self.__inits)
        ])
        start_omega = np.asarray([
            [scstats.dirichlet.rvs(alpha_z, 1, self.__rnd).squeeze() for alpha_z in self.__omega_init]
            for _ in range(self.__inits)
        ])

        # Run Multiple runs (in parallel):
        if self.__n_jobs == 0:  # Run serially
            self.__fit_params = [
                self.__fit_single(X, starts) for starts in zip(start_pi, start_psi, start_omega)
            ]
        else:  # Run in parallel using multiprocessing or threads
            mode = 'threads' if self.__n_jobs < 0 else 'processes'
            self.__fit_params = joblib.Parallel(n_jobs=abs(self.__n_jobs), prefer=mode)(
                joblib.delayed(self.__fit_single)(X, starts) for starts in zip(start_pi, start_psi, start_omega)
            )

        # Select best
        # Check if any converged
        if np.all([not fp['Converged'] for fp in self.__fit_params]):
            warnings.warn('None of the runs converged.')
        # Find run with maximum ll
        self.__best = np.argmax([fp['LLs'][-1] for fp in self.__fit_params])
        self.__pi = self.__fit_params[self.__best]['Pi'].copy()
        self.__psi = self.__fit_params[self.__best]['Psi'].copy()
        self.__omega = self.__fit_params[self.__best]['Omega'].copy()

        # Return Self
        return self

    def fit_partial(self, X, p_init=None):
        """
        Fits the Model from a warm-start

        Convenience method for fitting the data with a warm-start from the current parameters
        (or as supplied)

        :param X: The Observations to fit on: (see fit)
        :param p_init: The initialisation point for the parameters. Can be None, in which case,
                        the current values will be used.
        :return: self, for chaining
        """
        # Resolve initialisation
        if p_init is None:
            if (self.__pi is None) or (self.__psi is None) or (self.__omega is None):
                raise RuntimeError('One or more of the Parameters is not yet fit: you must supply initialiser')
            p_init = (self.Pi, self.Psi, self.Omega)
        else:
            p_init = (theta.copy(order='C') for theta in p_init)  # Ensure a copy

        # Fit the Data (run serially)
        self.__fit_params = [self.__fit_single(X, p_init)]

        # Check for Convergence (do not warn if doing one step)
        if not self.__fit_params[0]['Converged'] and self.__max_iter > 1:
            warnings.warn('None of the runs converged.')

        # Resolve Parameters and Store
        self.__best = 0 # There is only 1
        self.__pi = self.__fit_params[self.__best]['Pi'].copy()
        self.__psi = self.__fit_params[self.__best]['Psi'].copy()
        self.__omega = self.__fit_params[self.__best]['Omega'].copy()

        # Return Self
        return self

    def predict_proba(self, X):
        """
        Predicts probability over latent-state Z

        :param X: X-values: N list of [T, K, X].
        :return: probabilities over Z: N List of [T, Z]
        """
        return self.__responsibility(X, self.Pi, self.Psi, self.Omega, True)[0]

    def predict(self, X):
        """
        Argmax of predict proba. Note that this uses the MPM currently (Viterbi not implemented)

        :param X: X-values: N list of T x K x X.
        :return:
        """
        return [np.argmax(z, axis=1) for z in self.predict_proba(X)]

    def logpdf(self, X, per_run=False, norm=False):
        """
        Return the (evidence) log-likelihood for the data

        :param X: The observations to compute the evidence log-likelihood for: N-list of [T, K, X]
        :param per_run: If True, return the log-likelihood per each run individually
        :param norm: If True, normalise the log-likelihood by the number of emissions
        :return: Log-Likelihood: note that this does not include the prior likelihood
        """
        return self.__responsibility(X, self.Pi, self.Psi, self.Omega, True, per_run, norm)[-1]

    @property
    def Pi(self):
        return self.__pi.copy(order='C')

    @property
    def Psi(self):
        return self.__psi.copy(order='C')

    @property
    def Omega(self):
        return self.__omega.copy(order='C')

    @property
    def Evolution(self):
        return np.asarray(self.__fit_params[self.__best]['LLs'])

    @property
    def Stability(self):
        return np.asarray([fp['LLs'][-1] for fp in self.__fit_params])

    def __fit_single(self, X, p_init):
        """
        Private Method for fitting a single initialisation of the parameters using EM

        :param X: The Observations to fit on: see fit above
        :param p_init: Initial Value for the Pi/Psi/Omega probabilities
        :return: Dictionary with:
            * Pi: Fit Pi
            * Psi: Fit Psi
            * Omega: Fit Omega
            * LLs: Log-Likelihood evolution
            * Converged: True if converged, False otherwise
        """
        # Resolve Parameters
        pi, psi, omega = p_init
        dir_pi = scstats.dirichlet(self.__pi_prior)
        alpha_pi = self.__pi_prior - 1
        dir_psi = [[scstats.dirichlet(a_kz) for a_kz in a_k] for a_k in self.__psi_prior]
        alpha_psi = np.asarray([[a_kz - 1 for a_kz in a_k] for a_k in self.__psi_prior])
        dir_omega = [scstats.dirichlet(a_z) for a_z in self.__omega_prior]
        alpha_omega = np.asarray([a_z - 1 for a_z in self.__omega_prior])

        # Prepare to Run
        iters = 0
        log_likelihood = []

        # Run EM
        while iters < self.__max_iter:
            # ---- E-Step ---- #
            # 1) Compute Responsibilities
            gamma_pi, gamma_psi, eta_omega, ll = self.__responsibility(X, pi, psi, omega)
            # 2) Update Log-Likelihood, including contribution of (old) Pi/Psi/Omega
            ll += dir_pi.logpdf(pi)
            for k in range(self.sK):
                for z in range(self.sZ):
                    ll += dir_psi[k][z].logpdf(psi[k, z, :])
            for z in range(self.sZ):
                ll += dir_omega[z].logpdf(omega[z, :])
            log_likelihood.append(ll)

            # ---- Check for Convergence ---- #
            if self.__converged(log_likelihood):
                break

            # ---- M-Step ---- #
            pi = npext.sum_to_one(gamma_pi + alpha_pi)                  # 1) Update Pi
            psi = npext.sum_to_one(gamma_psi + alpha_psi, axis=-1)      # 2) Update Psi
            omega = npext.sum_to_one(eta_omega + alpha_omega, axis=-1)  # 3) Omega

            # ---- Iterations ---- #
            iters += 1

        # Return result
        return {
            'Pi': pi,
            'Psi': psi,
            'Omega': omega,
            'LLs': log_likelihood,
            'Converged': self.__converged(log_likelihood)
        }

    def __converged(self, lls):
        """
        Convergence Check

        :param lls: Array of Log-Likelihood
        :return: True only if converged, within tolerance
        """
        if len(lls) < 2:
            return False
        elif lls[-1] < lls[-2]:
            warnings.warn("Drop in Log-Likelihood Observed! Results are probably wrong.")
            return False
        else:
            return abs((lls[-1] - lls[-2]) / lls[-2]) < self.__tol

    def __responsibility(self, X, pi, psi, omega, posterior=False, ll_per_run=False, ll_norm=False):
        """
        Computes the responsibility summary statistics (Eqs 15 through 17)

        :param X: X (for all samples): N-list of sizes [T^n, K, X]. May contain NaNs
        :param pi: Pi Parameter (size [Z])
        :param psi: Psi Parameter (size [K, Z, X]
        :param omega: Omega Parameter (size [Z, Z])
        :param posterior: If True, return only the gamma matrix (posterior over latent states)
        :param ll_per_run: If True, return the log-likelihood per-run (instead of total)
        :param ll_norm: If True, normalise the log-likelihood by the Number of Emissions
        :return: Behaviour depends on setting of posterior:
            If True: two-tuple containing:
                * p_Z: posterior over each Z
                * ll: Log-likelihood
            else: four-tuple containing
                * gamma_pi: sufficient statistic for Pi
                * gamma_psi: sufficient statistic for Psi
                * eta_omega: sufficient statistic for Omega
                * ll: Log-likelihood
        """
        # Create Placeholders
        if posterior:
            p_Z = []
        else:
            gamma_pi = np.zeros_like(pi, order='C')
            gamma_psi = np.zeros_like(psi, order='C')
            eta_omega = np.zeros_like(omega, order='C')
        ll, n_obs = [], [] if ll_norm else None

        # Iterate over samples
        for X_n in X:
            # Pre-compute Sizes and fill NaNs
            sT = len(X_n)
            if ll_norm:
                n_obs.append(np.isfinite(X_n).all(axis=-1).sum())
            X_n = np.nan_to_num(X_n, nan=0.0)
            # Compute emission Probabilities
            P_X = np.empty([sT, self.sZ], order='C')
            self.__prob_emission(X_n, psi, P_X)
            # Compute Forward Pass
            F = np.empty([sT, self.sZ], order='C')
            C = np.empty(sT, order='C')
            self.__forward_single(P_X, pi, omega, F, C)
            # Compute Backward Pass
            B = np.empty([sT, self.sZ], order='C')
            self.__backward_single(P_X, omega, C, B)
            # Accumulate Sufficient Statistics
            if posterior:
                p_Z.append(F * B)
            else:
                gamma_pi += F[0, :] * B[0, :]  # Eq. 4/15
                self.__psi_single(X_n, F * B, gamma_psi)  # Eq 4/17
                self.__eta_single(P_X, omega, F, B, C, eta_omega)  # Eq 5/16
            # Accumulate LL
            ll.append(-np.log(C).sum())

        # Return, but need to resolve LL
        if ll_per_run:
            if ll_norm:
                ll = np.divide(ll, n_obs)
        else:
            ll = np.sum(ll)
            if ll_norm:
                ll /= np.sum(n_obs)
        return (p_Z, ll) if posterior else (gamma_pi, gamma_psi, eta_omega, ll)

    @staticmethod
    @njit(signature_or_function=(float64[:, :, :], float64[:, :, :], float64[:, :]))
    def __prob_emission(X_n, psi, P_X):
        """
        Compute P_X for a single sample (all time) (Eq 6)

        :param X_n: X for sample n: array-like of size [T, K, X]: note NaN's must be filled as 0s
        :param psi: Psi [K, Z, X]
        :param P_X: <output [T, Z]> P_X (Eq. 6)
        :return: None
        """
        # Find Sizes
        sT, sZ = P_X.shape

        # Compute over all T
        for t in range(sT):
            for z in range(sZ):
                P_X[t, z] = np.power(psi[:, z, :], X_n[t, :, :]).prod()

    @staticmethod
    @njit(signature_or_function=(float64[:,::1], float64[::1], float64[:,::1], float64[:,::1],
                                 float64[:]))
    def __forward_single(P_X, pi, omega, F_hat, C):
        """
        Compute the Forward Pass for a single sample (entire time). Eqs 7 through 9

        :param P_X: Emission evidence for state Z [T, Z]
        :param pi: Pi Matrix [Z]
        :param omega: Omega Matrix [Z, Z]
        :param F_hat: <output [T, Z]> Forward Pass Parameters (Eq 7)
        :param C: <output [T] > Normalisers (multiplier) (Eq 9)
        :return: None
        """
        # Compute some sizes
        sT, sZ = F_hat.shape

        # Do t=0
        F_hat[0, :] = pi * P_X[0, :]
        C[0] = 1.0/(F_hat[0, :].sum())
        F_hat[0, :] *= C[0]

        # Do t > 0
        for t in range(1, sT):
            for z in range(sZ):
                F_hat[t, z] = P_X[t, z] * (F_hat[t-1, :] * omega[:, z]).sum()
            C[t] = 1/(F_hat[t, :].sum())
            F_hat[t, :] *= C[t]

    @staticmethod
    @njit(signature_or_function=(float64[:, :], float64[:, :], float64[:], float64[:, :]))
    def __backward_single(P_X, omega, C, B_hat):
        """
        Compute the backward pass for a single sample (entire time). Eqs 10/11

        :param P_X: Emission evidence for state Z [T, Z]
        :param omega: Omega Matrix [Z, Z]
        :param C: Normalisers (multiplier) [T, Z]
        :param B_hat: <output [T, Z]> Backward Pass parameters (Eq. 10)
        :return: None
        """
        # Complete some sizes
        sT, sZ = B_hat.shape

        # t = T
        B_hat[-1, :] = 1

        # t < T
        for t in range(sT - 2, -1, -1):
            _pb = P_X[t+1, :] * B_hat[t+1, :]  # Last two terms do not depend on z
            for z in range(sZ):
                B_hat[t, z] = C[t+1] * (omega[z, :] * _pb).sum()

    @staticmethod
    @njit(signature_or_function=(
            float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:, :]
    ))
    def __eta_single(P_X, omega, F_hat, B_hat, C, eta):
        """
        Computes Eta's for a single sample (all time) Eq. 5/16

        :param P_X: Emission evidence for state Z: size [T, Z]
        :param omega: Omega Matrix size [Z, Z]
        :param F_hat: Forward Pass Parameters [T, Z]
        :param B_hat: Backward Pass Parameters [T, Z]
        :param C: Normalisers [T]
        :param eta: <output [Z, Z]> Sum of Eta responsibilities: should be initialised!
        :return: None
        """
        # Pre-Compute Sizes
        sT, sZ = P_X.shape

        # Compute Eta's
        for t in range(1, sT):
            for zp in range(sZ):
                for z in range(sZ):
                    eta[zp, z] += C[t] * F_hat[t-1, zp] * B_hat[t, z] * omega[zp, z] * P_X[t, z]

    @staticmethod
    @njit(signature_or_function=(float64[:, :, :], float64[:, :], float64[:, :, :]))
    def __psi_single(X, gamma, gamma_psi):
        """
        Accumulates (unnormalised) Psi sufficient statistic for a single sample (all time) Eq. 17
        :param X: Emission samples for a single n: NaNs must be filled as 0's [T, K, X]
        :param gamma: Gamma responsibilities [T, Z]
        :param gamma_psi: <output [K, Z, X]> Storage for Responsibilities (must be initialised)
        :return: None
        """
        # Pre-Compute Sizes
        sT, sK = X.shape[:2]
        sZ = gamma.shape[1]

        # Compute Stat
        for t in range(sT):
            for k in range(sK):
                for z in range(sZ):
                    gamma_psi[k, z, :] += gamma[t, z] * X[t, k, :]


def class_accuracy(y_true, y_pred, labels=None, normalize=True):
    """
    Computes per-class, one-v-rest accuracy

    :param y_true: True labels (N)
    :param y_pred: Predicted labels (N)
    :param labels: If not None, specifies labels to consider: otherwise any label that appears in
                   y_true or y_pred is considered
    :param normalize: If True, normalise relative to all samples: else report number of samples.
    :return: Accuracy-score per-class
    """
    # Define Labels
    labels = utils.default(labels, np.union1d(np.unique(y_pred), np.unique(y_true)))

    # compute per-class accuracy
    accuracy = np.empty(len(labels))
    for i, lbl in enumerate(labels):
        accuracy[i] = skmetrics.accuracy_score(y_true == lbl, y_pred == lbl, normalize=normalize)
    return accuracy


def multi_class_calibration(
        y_true, y_prob, n_bins=5, strategy='uniform', names=None, ref_line=True, ax=None, **kwargs
):
    """
    Displays a multi-class Calibration Curve (one line per class in one-v-rest setup)

    :param y_true: True Labels: note that labels must be sequential starting from 0
    :param y_prob: Predicted Probabilities
    :param n_bins: Number of bins to use (see CalibrationDisplay.from_predictions)
    :param strategy: Strategy for bins (see CalibrationDisplay.from_predictions)
    :param names:  Class names to use
    :param ref_line: Whether to plot reference line (see CalibrationDisplay.from_predictions)
    :param ax: Axes to draw on (see CalibrationDisplay.from_predictions)
    :param kwargs: Keyword arguments passed on to plot
    :return: Dict of Calibration Displays (by name)
    """
    # Iterate over classes
    names = utils.default(names, np.arange(y_prob.shape[1]))
    displays = {}
    for cls, name in zip(range(y_prob.shape[1]), names):
        _y_true = (y_true == cls).astype(int)  # Get positive class for this label
        _y_prob = y_prob[:, cls]  # Get probability assigned to this class
        displays[name] = CalibrationDisplay.from_predictions(
            _y_true, _y_prob, n_bins=n_bins, strategy=strategy, name=name, ref_line=ref_line, ax=ax,
            **kwargs
        )
    return displays


def hierarchical_log_loss(y_true, y_prob, mapping, eps=1e-15):
    """
    Compute the Log-Loss, when y_true contains over-arching labels which are not predictable in
    y_prob.

    :param y_true: The ground-truths
    :param y_prob: The predicted values - should be probabilities (or one-hot encoding)
    :param mapping: For each super-label, the set of label probabilities which must be summed. Must
                    be a dictionary of arrays, since will not start from 0. It is assumed that
                    the super-labels are contiguous and follow the fine-grained labels
                    immediately, which are themselves numbered zero through L-1
    :param eps:     A small value to avoid taking log of 0
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
        _ll[y_true == m_s] = np.multiply(
            y_prob[y_true == m_s, :], np.asarray(m_f)[np.newaxis, :]
        ).sum(axis=1)

    # Finally compute the actual log-loss (after clipping)
    return -np.sum(np.log(np.clip(_ll, eps, 1))) / _l


def multi_way_split(y, sizes, splitter, random_state=None):
    """
    Convenience Function for wrapping a multi-way split. This only returns the indices of the split.
    This is actually implemented as a recursive function.

    :param y:  The target labels. If using a stratified splitter, then this must be the true
               targets: otherwise, it is enough to be an empty array of the same length as the data.
    :param sizes:  The relative sizes of the K sets. Note that these should sum to 1 (this will
                   be enforced by dividing by the sum).
    :param splitter:  The splitting object: this allows stratified/unstratified type splits
                      (basically one of ShuffleSplit or StratifiedShuffleSplit). Note that you
                      should NOT initialise the object: this is just passing a reference to the
                      class (and not an object).
    :param random_state:  Any random state to employ
    :return:  K-tuple of indices, one each for the K sets.
    """
    # --- In either case, ensure that the sizes sum to 1! --- #
    sizes = npext.sum_to_one(sizes)
    # --- Base Case: We know how to do this --- #
    if len(sizes) == 2:
        return next(
            splitter(
                n_splits=1, train_size=sizes[0], test_size=sizes[1], random_state=random_state,
            ).split(y, y)
        )
    # --- Other Cases --- #
    #   This is a bit trickier. We have to first split assuming that all but the first set are
    #   grouped together. We then pass the second set of targets recursively to our function,
    #   with the remaining sizes. However, when the indices are returned, they must be remapped
    #   to the original index set, since they are indices into that set. Also, to ensure
    #   randomness, the seed is increased by one each time.
    sub_sizes = sizes[1:]
    left, right = next(
        splitter(
            n_splits=1, train_size=sizes[0], test_size=np.sum(sub_sizes), random_state=random_state,
        ).split(y, y)
    )
    right_split = multi_way_split(
        y[right], sub_sizes, splitter, random_state + 1 if random_state is not None else None,
    )
    idcs = [left]
    for i in right_split:
        idcs.append(right[i])
    return idcs


def net_benefit_curve(y_true, y_score, pos_label=1, epsilon=1e-3):
    """
    Creates a Net-Benefit curve for various thresholds t.

    It basically replicates sklearn.metrics.roc_curve

    :param y_true:  Groundtruth labels
    :param y_score: Classifier scores
    :param pos_label: Which label to treat as positive
    :param epsilon: replacement for 0 - threshold
    :return:
    """
    # Get Statistics
    N, pos, neg = len(y_true), (y_true == pos_label).sum(), (y_true != pos_label).sum()

    # Compute Rates at different thresholds and subsequently net benefit
    fpr, tpr, thr = skmetrics.roc_curve(y_true, y_score, pos_label=pos_label)
    fpr, tpr, thr = np.flip(fpr)[:-1], np.flip(tpr)[:-1], np.flip(np.clip(thr, 0, 1-epsilon))[:-1]
    nb = (tpr * pos / N) - (fpr * neg / N) * (thr / (1 - thr))

    # Return
    return nb, thr


def mlp_complexity(mlp):
    """
    Computes the complexity (number of trainable parameters) of a MLP model

    :param mlp: Model to evaluate. This must have been fit (or at least seen some data)
    :return: Number of (scalars) to learn
    """
    return np.sum([np.prod(c.shape) for c in mlp.coefs_]) + np.sum([np.prod(i.shape) for i in mlp.intercepts_])


class HierarchicalClustering:
    """
    A Class to wrap Scipy's Linkage methods in a convenient framework similar to sklearn. This adds
    some flexibility to SKLearn's own AgglomerativeClustering, for example in visualising
    dendrograms.
    """

    def __init__(self, n_clusters=2, affinity="euclidean", link_type="ward"):
        """
        Initialise the Clustering

        :param n_clusters: Number of Clusters. Due to the nature of the algorith,, this can
                           technically be updated later without having to retrain the model.
        :param affinity: The type of distance metric to use. By default this is the euclidean
                         metric: and in general accepts all the metrics defined in
                         scipy.spatial.distance.pdist (see https://docs.scipy.org/doc/scipy/
                         reference/generated/scipy.spatial.distance.pdist.html).
                         In addition, there is the option to pass a precomputed distance matrix,
                         in which case, this should be 'precomputed'.
        :param link_type: Which linkage criterion to use. The linkage criterion determines which
                          distance to use between sets of observation. The algorithm will merge
                          the pairs of cluster that minimize this criterion. (See the
                          documentation for scipy's linkage method https://docs.scipy.org/doc/scipy/
                          reference/generated/scipy.cluster.hierarchy.linkage.html).
        """
        self.__n_clusters = n_clusters
        self.__affinity = affinity.lower()
        self.__linkage = link_type.lower()
        self.__clusters = None

    @property
    def NClusters(self):
        return self.__n_clusters

    @NClusters.setter
    def NClusters(self, n_clusters):
        """
        Allow Setting the Number of Clusters dynamically
        :param n_clusters: Number of clusters.
        :return:
        """
        if n_clusters > 0:
            self.__n_clusters = int(n_clusters)
        else:
            raise ValueError("Value must be an integer greater than 0")

    def fit(self, X, y=None):
        """
        Fit a Hierarchical Clustering Scheme to the Data X

        :param X: If the affinity metric was set to 'precomputed', this must be a precomputed
                  distance matrix, of size N x N, where entry X_{i,j} is the distance between
                  sample i and sample j. Otherwise it is a 2D array of size N x M where M is the
                  feature-space size.
        :param y: ignored, but provided for compatibility with fit
        :return: self, for chaining.
        """
        # If Precomputed, convert to Condensed Form first
        if self.__affinity == "precomputed":
            X = squareform(X, checks=False)
        # Cluster
        self.__clusters = linkage(
            y=X, method=self.__linkage, metric=self.__affinity, optimal_ordering=True
        )
        # Return self for chaining
        return self

    def predict(self, X=None):
        """
        Predict the pre-trained labels. Note that this can be called after changing the number of
        clusters even without rerunning fit.

        :param X: Ignored. This is provided for convenience. Indeed, the clustering can only be done
                  relative to the original matrix.
        :return:  Cluster Labels for each sample (labelled 0 to NClusters-1)
        """
        return fcluster(self.__clusters, t=self.__n_clusters, criterion="maxclust")

    def fit_predict(self, X, y):
        """
        Convenience method to fit and then predict the cluster labels. Refer to fit/predict for
        explanation.

        :param X: If the affinity metric was set to 'precomputed', this must be a precomputed
                  distance matrix, of size N x N, where entry X_{i,j} is the distance between
                  sample i and sample j. Otherwise it is a 2D array of size N x M where M is the
                  feature-space size.
        :param y: ignored
        :return:  Cluster Labels for each sample (labelled 0 to NClusters-1)
        """
        return self.fit(X).predict()

    def plot_dendrogram(self, ax=None, labels=None, x_rot=0, color=None, fs=None):
        """
        Plot a Dendrogram of the Agglomeration procedure

        :param ax:      Axes to plot on: if not specified, then uses a new axis.
        :param labels:  Labels for the leaf nodes: if not specified, will just number 0 through N-1
        :param x_rot:   Rotation for the leaf node text
        :param color:   Threshold to use for colouring. If None, then do not differ in colours
                        (similar to negative): else it signifies a threshold to use as per:
                        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster
                        .hierarchy.dendrogram.html)
        :param fs:      Font-size for plotting the labels
        :return:        self, for chaining
        """
        if color is None:
            color = -1
        dendrogram(
            self.__clusters,
            ax=ax,
            labels=labels,
            leaf_rotation=x_rot,
            color_threshold=color,
            leaf_font_size=fs,
        )
        return self


class LogitCalibrator(tnn.Module):
    """
    Implementation of Temperature scaling which conforms to sklearn framework
    """
    def __init__(self, classes=None, theta_init=1.0, lr=1e-4, max_iter=100):
        """
        Initialises the Model
        :param theta_init: Initial value for scaling parameter
        :param lr:  Learning rate
        :param max_iter: Maximum number of iterations
        """
        # Call BaseClass
        super(LogitCalibrator, self).__init__()

        # Initialise some Parameters
        self.theta = theta_init
        self.__lr = lr
        self.__max_iter = max_iter
        self.classes_ = classes

        # Torch requirements
        self._theta = tnn.Parameter(torch.tensor([theta_init], dtype=torch.double))

    def fit(self, X, y):
        """
        Fits the model on the training Data

        :param X: The input logits
        :param y: The output labels (one of L behaviours, 0-indexed)
        """
        # Update Class List
        if self.classes_ is None:
            self.classes_ = np.unique(y)
        assert X.shape[1] == len(self.classes_), f'# Classes ({len(self.classes_)}) does not equal the size of the Logits ({X.shape[1]})'

        # Start by transforming to tensors.
        X, y = torch.tensor(X, dtype=torch.double), torch.tensor(y, dtype=torch.long)

        # Define optimiser (and closure for it)
        self.train()
        optimiser = torch.optim.LBFGS(self.parameters(), lr=self.__lr, max_iter=self.__max_iter)
        loss_func = tnn.NLLLoss()

        def _optim_step():
            optimiser.zero_grad()
            loss = loss_func(self(X).log(), y)
            loss.backward()
            return loss

        # Optimise Model
        optimiser.step(_optim_step)

        # Now get the parameters of interest
        self.eval()
        with torch.no_grad():
            self.theta = self._theta.numpy()[0]

        # Return self for chaining
        return self

    def predict_proba(self, X):
        """
        Predict Probabilities
        """
        return softmax(X / self.theta, axis=1)

    def predict(self, X):
        """
        Predict Behaviour

        Note, this is really a dummy, since the ordering does not change.
        """
        return np.argmax(X, axis=1)

    def forward(self, x):
        """
        Internal function for optimisation only.

        Computes a forward pass on x (a tensor of logits, of size [# Samples, # Labels])
        """
        return (x / self._theta).softmax(dim=1)


# # Test for MixtureOfCategoricals
# if __name__ == '__main__':
#
#     import time as tm
#
#     # Base Params
#     rng = np.random.default_rng(5)
#     sZ = 7
#     prior_beh = np.asarray([0.51369704, 0.09231234, 0.00780774, 0.05348966, 0.01216916, 0.02374176, 0.29678229])
#     nmdl = np.asarray(
#         [[6.9915, 0.2927, 0.2693, 0.3327, 0.3154, 0.306, 0.6544],
#          [0.2653, 2.3654, 0.3171, 0.3109, 0.3093, 0.2894, 0.6172],
#          [0.2386, 0.3238, 4.6861, 0.2981, 0.2966, 0.2897, 0.4505],
#          [0.5156, 0.282, 0.2516, 1.3101, 0.3511, 0.299, 0.6347],
#          [0.6059, 0.2882, 0.2894, 0.4569, 1.0259, 0.3024, 0.9283],
#          [0.3673, 0.428, 0.3601, 0.3904, 0.55, 3.2823, 2.6791],
#          [0.4415, 0.402, 0.3436, 0.4814, 0.596, 0.8654, 3.2864]]
#     )
#
#     # Generate Samples
#     print('Sampling ...')
#     pi = scstats.dirichlet.rvs(np.ones(sZ), 1, rng).squeeze()  # Sample Pi
#     psi = scstats.dirichlet.rvs(prior_beh + 1, [3, sZ], rng)  # Sample Psi
#     _, X = MixtureOfCategoricals(pi, psi, random_state=rng).sample(50000, as_probs=True, noisy=nmdl)
#
#     # Now try to fit the model, starting from correct point
#     print('Fitting Model')
#     s = tm.time()
#     model = MixtureOfCategoricals(sZ, [3, 7], inits=1, max_iter=100, n_jobs=0, random_state=rng)
#     res = model.__fit_single(X, p_init=(pi, psi))
#     model.fit(X)
#     print('Duration = ', tm.time() - s)
#
#     print(model.Pi)
#     print(model.Psi)

# Test for CHMM
if __name__ == '__main__':

    from scipy.spatial import distance as scdist
    import time as tm

    # rnd = np.random.default_rng(10)
    # NT = [10000, 8000, 12000, 5000, 1000, 4000, 2000, 10000, 4000, 6000]
    #
    # pi = np.asarray([0.3, 0.5, 0.2])
    # psi = np.asarray([
    #     [
    #         [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1],
    #         [0.1, 0.1, 0.3, 0.3, 0.05, 0.1, 0.05],
    #         [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]
    #     ],
    #     [
    #         [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1],
    #         [0.1, 0.1, 0.3, 0.3, 0.05, 0.1, 0.05],
    #         [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]
    #     ]
    #
    # ])
    # omega = np.asarray([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.2, 0.2, 0.6]])
    #
    # print('Sampling ... ', end=''); s = tm.time()
    # _, X = CategoricalHMM(pi, psi, omega,random_state=rnd).sample(NT, True, None)
    # print(f' Done! [{utils.show_time(tm.time() - s)}]')
    #
    # print('Learning Model (from init) ... ', end=''); s = tm.time()
    # fit = CategoricalHMM(pi, psi, omega, random_state=rnd, n_jobs=0).__fit_single(X)
    # print(f' Done! [{utils.show_time(tm.time() - s)}]')
    # print(pi - fit['Pi'])
    # print(psi - fit['Psi'])
    # print(omega - fit['Omega'])
    # print(fit['LLs'])

    # Create Random Generator (will be used throughout) and other common data
    A1_SAMPLES_N = 5  # This is Fixed
    A1_SAMPLES_T = np.asarray((0.1, 0.5, 1)) * 1000  # Number of samples (in thousands)
    A1_PSI_SAMPLER = (0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.35)
    A1_OMEGA_SAMPLER = [20, 1]  # Sampler for on-state/off-state alphas
    MAX_ITER = 200
    LOG_BASE = 2

    replica = 0; sZ = 2
    rng = np.random.default_rng(int(11 + replica + sZ * 100))
    iters, pi_errors, psi_errors, omega_errors = (np.empty(len(A1_SAMPLES_T)) for _ in range(4))

    # Generate Model and Sample from it
    # --- Generate Model --- #
    pi = scstats.dirichlet.rvs(np.ones(sZ), 1, rng).squeeze()  # Sample Pi
    psi = scstats.dirichlet.rvs(A1_PSI_SAMPLER, [3, sZ], rng)  # Sample Psi
    omega = np.empty([sZ, sZ])  # Sample Omega
    omega_sampler = np.ones(sZ) * A1_OMEGA_SAMPLER[1]
    omega_sampler[0] = A1_OMEGA_SAMPLER[0]
    for zz in range(sZ):
        omega[zz, :] = scstats.dirichlet.rvs(np.roll(omega_sampler, zz), 1, rng)
    # --- Generate Data --- #
    _, X = CategoricalHMM(pi, psi, omega, random_state=rng).sample(
        [int(A1_SAMPLES_T[-1])] * A1_SAMPLES_N, as_probs=True, noisy=None)

    # Now for each size
    for i, sT in enumerate(A1_SAMPLES_T):
        # 1) Train Model
        fit = CategoricalHMM(pi, psi, omega, max_iter=MAX_ITER).__fit_single([x[:int(sT), :, :] for x in X])
        # 2) Compute Errors/Stats
        iters[i] = len(fit['LLs'])
        pi_errors[i] = np.square(scdist.jensenshannon(pi, fit['Pi'], base=LOG_BASE))
        psi_errors[i] = np.square(
            scdist.jensenshannon(psi, fit['Psi'], base=LOG_BASE, axis=-1)).mean()
        omega_errors[i] = np.square(
            scdist.jensenshannon(omega, fit['Omega'], base=LOG_BASE, axis=-1)).mean()
