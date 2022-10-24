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
from sklearn.svm import SVC
import numpy as np


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
