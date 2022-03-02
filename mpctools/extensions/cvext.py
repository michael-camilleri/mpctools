"""
This Module will serve as an alternative and extension to opencv - hence the name

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
import glob

from numba import jit, uint8, uint16, double
from queue import Queue, Empty, Full
from scipy import optimize as optim
from threading import Thread
from scipy import linalg
import numpy as np
import time as tm
import warnings
import math
import cv2
import os


# Define Some Constants
VP_CUR_PROP_POS_MSEC = -100  # Current frame (rather than next one) in MS
VP_CUR_PROP_POS_FRAMES = -101  # Current frame (rather than next one) in index


class FourCC:
    """
    FourCC Wrapper to allow conversions between Integer and String Representation
    """

    @classmethod
    def to_int(cls, fourcc) -> int:
        return int(cv2.VideoWriter_fourcc(*fourcc))

    @classmethod
    def to_str(cls, fourcc) -> str:
        return "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])


class Homography:
    """
    Class for fitting a Homography:

    Note that when fitting, the class gives priority to the forward transformation (world to
    image). The image to world mapping is simply the inverse of the forward mapping, normalised
    such that the last entry, H_{3, 3} = 1
    """

    MTHD_OPENCV_LS = 0
    MTHD_OPENCV_RNSC = 1
    MTHD_OWN = 10

    @staticmethod
    def __convert(points, homog):
        points = np.array(points, copy=False, ndmin=2)
        valid = np.isfinite(points).all(axis=1)
        new_coords = np.full_like(points, fill_value=np.NaN)
        new_coords[valid, :] = np.squeeze(
            cv2.perspectiveTransform(np.expand_dims(points[valid, :], axis=0), homog)
        )
        return new_coords

    @staticmethod
    def __cost(h, w, i):
        x_err = np.square(
            i[:, 0]
            - (w[:, 0] * h[0] + w[:, 1] * h[1] + h[2]) / (w[:, 0] * h[6] + w[:, 1] * h[7] + 1)
        ).sum()
        y_err = np.square(
            i[:, 1]
            - (w[:, 0] * h[3] + w[:, 1] * h[4] + h[5]) / (w[:, 0] * h[6] + w[:, 1] * h[7] + 1)
        ).sum()
        return x_err + y_err

    def __init__(self, image_coords, world_coords, method=MTHD_OPENCV_LS, params=None):
        """
        Initialiser

        :param image_coords: A 2D Numpy array of image coordinates, with x/y along the second axis.
                             Must be of length at least 4.
        :param world_coords: A 2D Numpy array of corresponding world-coordinates: must be same shape
                            as image_coords
        :param method: An indication of which method to use for fitting:
                0: Use OpenCV's LeastSquares method
                1: Use OpenCV's RANSAC: in this case, params MUST specify the inlier cutoff (in
                terms of pixel distance) as a scalar.
                10: Use my method based on Nelder-Mead and setting H_{3,3} = 1
        :param params: Parameters needed for each of the methods
                MTHD_OPENCV_LS: ignored
                MTHD_OPENCV_RNSC: Inlier cutoff, scalar
                MTHD_OWN: Starting Solution (initial Homography Matrix)
        """
        # Forward Mapping
        if method == self.MTHD_OPENCV_LS:
            self.toImg = cv2.findHomography(world_coords, image_coords, method=0)[0]
            self.res = None
        elif method == self.MTHD_OPENCV_RNSC:
            if params is None:
                raise ValueError("Method MTHD_OPENCV_RNSC must specify the Inlier Cutoff.")
            self.toImg, self.res = cv2.findHomography(
                world_coords, image_coords, cv2.RANSAC, params
            )
        elif method == self.MTHD_OWN:
            if params is None:
                raise ValueError("Method MTHD_OWN needs the starting point.")
            self.res = optim.minimize(
                self.__cost,
                params.flatten()[:-1],
                args=(world_coords, image_coords),
                method="Nelder-Mead",
                options={"maxiter": 10000},
            )
            self.toImg = np.append(self.res.x, 1).reshape(3, 3)
        else:
            raise ValueError("Method must be one of MTHD_OPENCV_LS, MTHD_OPENCV_RNSC or MTHD_OWN")
        # Reverse Mapping
        self.toWrld = np.linalg.inv(self.toImg)
        self.toWrld /= self.toWrld[2, 2]

    def to_image(self, points):
        """
        Convert world coordinates to image coordinates

        :param points: 2D Numpy array, with the last dimension of size 2 (X/Y coordinates): if 1D
                       will be automatically promoted to 2D
        :return:    Image Coordinates
        """
        return self.__convert(points, self.toImg)

    def to_world(self, points):
        """
        Convert Image coordinates to world-coordinates

        :param points: 2D Numpy array, with the last dimension of size 2 (X/Y coordinates): if 1D
                       will be automatically promoted to 2D
        :return:    World Coordinates
        """
        return self.__convert(points, self.toWrld)


class AffineTransform:
    """
    Defines an Affine Transform.

    This class is an extension to skimage's and opencv's affine transformation. It encapsulates the
    definition of an affine, storing its parameters, and implements fitting the parameters in
    such a way that one can use both points and/or line correspondences.

    Note that as regards the individual components, we define an affine transformation by a
    translation (t_x, t_y) plus a scale (s_x, s_y), shear (m) and rotation (θ), in that order, st:

       | a  b  c |   | cos(θ)  -sin(θ)  0 | | 1  m  0 | | s_x  0  0 |     | 0  0  t_x |
       | d  e  f | = | sin(θ)   cos(θ)  0 | | 0  1  0 | |  0  s_y 0 |  +  | 0  0  t_y |
       | 0  0  1 |   |   0        0     0 | | 0  0  0 | |  0   0  0 |     | 0  0   1  |

    This is different from how skimage defines the components, and has implications on how the
    decomposition happens.

    Note also, that by definition, vectors are defined as [X, Y (1)]
    """

    AFFINE = 6
    SIMILARITY = 4
    TRANSLATION = 2

    @staticmethod
    def characterise(matrix):
        """
        Characterises the Transform, according to Stephan's Answer in:
             https://math.stackexchange.com/questions/612006/decomposing-an-affine-transformation
        :param matrix: The Affine Transform
        :return: Translation, Scaling, Shear, Rotation
        """
        b = matrix[:2, -1]
        s_x = np.linalg.norm(matrix[:2, 0])
        theta = np.arctan2(matrix[1, 0], matrix[0, 0])

        _sin = np.sin(theta)
        _cos = np.cos(theta)

        msy = matrix[0, 1] * _cos + matrix[1, 1] * _sin
        if _sin != 0:
            s_y = (msy * _cos - matrix[0, 1]) / _sin
        else:
            s_y = (matrix[1, 1] - msy * _sin) / _cos

        m = msy / s_y

        return b, np.asarray((s_x, s_y)), m, theta

    def __init__(self, matrix=None, scale=(1, 1), rotation=0, shear=0, translation=(0, 0)):
        """
        Initialises the Model, using either the matrix or any of the other parameters.

        :param matrix: Matrix description of the Affine Transform. If this is specified,
            then it takes precedence in the specification of the transform. Note that it can be
            specified in either augmented euclidean [2x3] or homogenous [3x3] form (in the latter
            case, the last row is completely ignored, and no check is done for correctness).
        :param scale: Scale parameter. 2D (X/Y), or scalar (same)
        :param rotation: Rotation parameter (theta) in *radians*
        :param shear: Shear parameter
        :param translation: Translation. 2D (X/Y) or scalar (same)
        """
        if matrix is not None:
            # Handle Shape
            if matrix.shape[0] == 3:
                self._forward = matrix[:2, :]
            else:
                self._forward = matrix
            # Characterise
            self._params = self.characterise(self._forward)
        else:
            sin_r, cos_r = np.sin(rotation), np.cos(rotation)
            sx, sy = (scale, scale) if np.isscalar(scale) else scale
            tx, ty = (translation, translation) if np.isscalar(translation) else translation
            self._forward = np.asarray(
                [
                    [sx * cos_r, sy * (shear * cos_r - sin_r), tx],
                    [sx * sin_r, sy * (shear * sin_r + cos_r), ty],
                ]
            )
            self._params = (np.asarray((tx, ty)), np.asarray((sx, sy)), shear, rotation)
        # Compute Inverse once
        try:
            self._inverse = np.linalg.inv(np.append(self._forward, [[0, 0, 1]], axis=0))[:2, :]
        except np.linalg.LinAlgError:
            self._inverse = None
            warnings.warn("Forward Transform is Singular and cannot be inverted.")
        # Finally, for estimation, we reserve the quality
        self._qlt = None

    def __repr__(self):
        _s = [f'{s:.2f}' for s in self.scale]
        _t = [f'{t:.0f}' for t in self.translation]
        return f'AFFINE {{R({self.rotation:.2f}) M({self.shear:.2f}) S{_s} T{_t}}}'

    def estimate(self, pts=None, lns=None, weight=0.5, dof=AFFINE):
        """
        Estimates the transform from point and/or line correspondences

        The method can estimate the affine transformation from either point or line
        correspondences (or both). Points themselves may be specified in either euclidean or
        homogeneous co-ordinates, yielding an Nx(2/3) matrix. Lines are specified in terms of two
        (end-)points (each euclidean or homogeneous), and hence is an Nx2x(2/3) matrix.

        :param pts: Source/Destination Points. 2-Tuple of array-like or None
        :param lns: Destination Points/Lines. 2-Tuple of Arraylike
        :param weight: Weight for Point (as opposed to line) correspondences.
        :param dof: Degrees of Freedom (in decreasing flexibility)
                    AFFINE: Full affine [Default]
                    SIMILARITY: Similarity Transform (univariate scale, rotation and translation)
                    TRANSLATION: Translation-only (2D)
        :return: self, for chaining
        """
        # Set up Weights
        pt_w, ln_w = np.sqrt(weight), np.sqrt(1 - weight)
        # Some Error checking
        if pts is None and lns is None:
            raise ValueError("You must specify at least one of either point/line correspondences.")
        if dof not in (self.AFFINE, self.SIMILARITY, self.TRANSLATION):
            raise ValueError("DOF Mode not recognised.")
        if pts is not None:
            if len(pts) != 2:
                raise ValueError("You must specify both Source and Destination Points")
            if len(pts[0]) != len(pts[1]):
                raise ValueError("Destination and Source Points must be same size.")
        else:
            pts = (None, None)
            ln_w = 1  # Update line weight to be 1
        if lns is not None:
            if len(lns) != 2:
                raise ValueError("You must specify both Source and Destination Lines")
            if len(lns[0]) != len(lns[1]):
                raise ValueError("Destination and Source Lines must be same size.")
        else:
            lns = (None, None)
            pt_w = 1  # Update point weight to be 1

        # Get Normalised Point/Line coordinates
        (n_p, n_l), (x_s, y_s), (u_s, v_s, w_s), T_s = self.__normalise(pts[0], lns[0], dof == self.TRANSLATION)
        _, (x_d, y_d), (u_d, v_d, w_d), T_d = self.__normalise(pts[1], lns[1], dof == self.TRANSLATION)

        # Fill up:
        #  Note that for simplicity, I am not interlacing, instead opting to stack
        #  - Start with e -
        e = np.zeros([(n_p + n_l) * 2, 1], dtype=np.float64)
        if n_p > 0:
            if dof == self.TRANSLATION:
                e[:n_p, 0] = (x_d - x_s) * pt_w
                e[n_p: n_p * 2, 0] = (y_d - y_s) * pt_w
            else:
                e[:n_p, 0] = x_d * pt_w
                e[n_p : n_p * 2, 0] = y_d * pt_w
        if n_l > 0:
            if dof == self.TRANSLATION:
                e[n_p * 2: n_p * 2 + n_l, 0] = (w_s * v_d - v_s * w_d) * ln_w
                e[n_p * 2 + n_l:, 0] = (u_s * w_d - w_s * u_d) * ln_w
            else:
                e[n_p * 2 : n_p * 2 + n_l, 0] = -v_s * w_d * ln_w
                e[n_p * 2 + n_l :, 0] = u_s * w_d * ln_w
        #  - Now, F -
        if dof == self.TRANSLATION:
            F = np.zeros([(n_p + n_l) * 2, 2], dtype=np.float64)
            if n_p > 0:
                F[:n_p, 0] = pt_w
                F[n_p : n_p * 2, 1] = pt_w
            if n_l > 0:
                F[n_p * 2: n_p * 2 + n_l, 0] = v_s * u_d * ln_w
                F[n_p * 2: n_p * 2 + n_l, 1] = v_s * v_d * ln_w
                F[n_p * 2 + n_l :, 0] = -u_s * u_d * ln_w
                F[n_p * 2 + n_l :, 1] = -u_s * v_d * ln_w
        elif dof == self.SIMILARITY:
            F = np.zeros([(n_p + n_l) * 2, 4], dtype=np.float64)
            if n_p > 0:
                F[:n_p, 0] = x_s * pt_w
                F[:n_p, 1] = -y_s * pt_w
                F[:n_p, 2] = pt_w
                F[n_p: n_p * 2, 0] = y_s * pt_w
                F[n_p: n_p * 2, 1] = x_s * pt_w
                F[n_p: n_p * 2, 3] = pt_w
            if n_l > 0:
                F[n_p * 2: n_p * 2 + n_l, 0] = -w_s * v_d * ln_w
                F[n_p * 2: n_p * 2 + n_l, 1] = w_s * u_d * ln_w
                F[n_p * 2: n_p * 2 + n_l, 2] = v_s * u_d * ln_w
                F[n_p * 2: n_p * 2 + n_l, 3] = v_s * v_d * ln_w
                F[n_p * 2 + n_l:, 0] = w_s * u_d * ln_w
                F[n_p * 2 + n_l:, 1] = w_s * v_d * ln_w
                F[n_p * 2 + n_l:, 2] = -u_s * u_d * ln_w
                F[n_p * 2 + n_l:, 3] = -u_s * v_d * ln_w
        elif dof == self.AFFINE:
            F = np.zeros([(n_p + n_l) * 2, 6], dtype=np.float64)
            if n_p > 0:
                F[:n_p, 0] = x_s * pt_w
                F[:n_p, 1] = y_s * pt_w
                F[:n_p, 2] = pt_w
                F[n_p : n_p * 2, 3] = x_s * pt_w
                F[n_p : n_p * 2, 4] = y_s * pt_w
                F[n_p : n_p * 2, 5] = pt_w
            if n_l > 0:
                F[n_p * 2 : n_p * 2 + n_l, 1] = -w_s * u_d * ln_w
                F[n_p * 2 : n_p * 2 + n_l, 2] = v_s * u_d * ln_w
                F[n_p * 2 : n_p * 2 + n_l, 4] = -w_s * v_d * ln_w
                F[n_p * 2 : n_p * 2 + n_l, 5] = v_s * v_d * ln_w
                F[n_p * 2 + n_l :, 0] = w_s * u_d * ln_w
                F[n_p * 2 + n_l :, 2] = -u_s * u_d * ln_w
                F[n_p * 2 + n_l :, 3] = w_s * v_d * ln_w
                F[n_p * 2 + n_l :, 5] = -u_s * v_d * ln_w

        # Solve, and also build Isotropic Transform
        m, res, rnk, s = linalg.lstsq(
            F, e, overwrite_a=True, overwrite_b=True, lapack_driver="gelss"
        )

        # Build Result
        if dof == self.TRANSLATION:
            M = np.asarray([[1, 0, m[0]], [0, 1, m[1]], [0, 0, 1]], dtype=np.float64)
        elif dof == self.SIMILARITY:
            M = np.asarray([[m[0], -m[1], m[2]], [m[1], m[0], m[3]], [0, 0, 1]], dtype=np.float64)
        else:
            M = np.asarray([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0, 0, 1]], dtype=np.float64)
        self._forward = (np.linalg.inv(T_d) @ M @ T_s)[:2, :]
        self._params = self.characterise(self._forward)
        try:
            self._inverse = np.linalg.inv(np.vstack([self._forward, [[0, 0, 1]]]))[:2, :]
        except np.linalg.LinAlgError:
            self._inverse = None
            warnings.warn("Forward Transform is Singular and cannot be inverted.")
        self._qlt = (res.sum(), rnk, abs(s[0] / s[-1]))

        # Return self for chaining
        return self

    def forward(self, points):
        """
        Transforms the points in the forward direction
        """
        return self.__apply_transform(self._forward, points)

    def inverse(self, points):
        """
        Transforms the points ine reverse direction
        """
        if self._inverse is not None:
            return self.__apply_transform(self._inverse, points)
        else:
            raise RuntimeError("No Inverse exists for this Transform.")

    @property
    def fit_quality(self):
        if self._qlt is not None:
            return {"Residuals": self._qlt[0], "Rank": self._qlt[1], "Condition": self._qlt[2]}
        else:
            return None

    @property
    def matrix_f(self):
        return self._forward.copy()

    @property
    def matrix_i(self):
        return self._inverse.copy()

    @property
    def translation(self):
        return np.copy(self._params[0])

    @property
    def scale(self):
        return np.copy(self._params[1])

    @property
    def shear(self):
        return self._params[2]

    @property
    def rotation(self):
        return self._params[3]

    @staticmethod
    def __normalise(pts, lines, translate_only=False):
        """
        Gets Normalised Points/Line Entries

        Note that a line is defined in terms of its end-point. If translate_only is set,
        then there is no scaling, just translation

        Returns Normalised Parameters
            (p, l): Number of Points/Lines respectively
            (x, y): If points provided, normalised x/y coordinates
            (u, v, w): If lines provided, normalised u/v/w coordinates
            T: Translation/Isotropic Scaling matrix
        """
        # Build Points
        all_pts = []
        if pts is not None:
            pts = np.array(pts, dtype=np.float64, copy=True)
            if pts.shape[1] == 3:
                pts /= pts[:, [2]]
            all_pts.append(pts[:, :2]) #
        if lines is not None:
            lines = np.array(lines, dtype=np.float64, copy=True)
            if lines.shape[2] == 3:
                lines = lines[:, :, :2] / lines[:, :, [2]]
            all_pts.append(lines[:, 0, :])  # Begin
            all_pts.append(lines[:, 1, :])  # End
        all_pts = np.vstack(all_pts)

        # Find (forward) Transform [i.e. Add/Multiply)
        t = -all_pts.mean(axis=0)
        if translate_only:
            s = 1
        else:
            s = np.sqrt(2) / np.linalg.norm(all_pts + t, axis=1).mean()
        T = np.asarray([[s, 0, t[0] * s], [0, s, t[1] * s], [0, 0, 1]], dtype=np.float64)

        # Prepare Values to Return
        if pts is not None:
            x, y = (pts[:, 0] + t[0]) * s, (pts[:, 1] + t[1]) * s
            p = len(x)
        else:
            x, y = None, None
            p = 0
        if lines is not None:
            l_b = (lines[:, 0, :] + t[np.newaxis, :]) * s
            l_e = (lines[:, 1, :] + t[np.newaxis, :]) * s
            u, v, w = np.apply_along_axis(build_line, 1, np.hstack([l_b, l_e])).T
            l = len(u)
        else:
            u, v, w = None, None, None
            l = 0

        # Return
        return (p, l), (x, y), (u, v, w), T

    @staticmethod
    def __apply_transform(matrix: np.ndarray, points: np.ndarray):
        """
        Applies a Matrix Transformation on a set of points. Takes care of converting them to 2D
        if need be and ignoring. Note that the returned value is always in R^2, and will be
        squeezed if there is only one point.
        :return:
        """
        # Ensure first that the coordinates are a 2D array
        points = np.array(points, copy=False, ndmin=2)
        # If need be, add 1 for homogeneous coordinate
        if points.shape[1] == 2:
            points = np.append(points, np.ones([points.shape[0], 1]), axis=1)
        elif points.shape[1] == 3:
            points = points / points[:, -1][:, np.newaxis]
        else:
            raise ValueError("points parameter must live in R^2 or R^3.")
        # Perform Multiplication (and squeeze out redundant dimensions)
        return np.squeeze(points @ matrix.T)


def pairwise_iou(a, b, cutoff=0, distance=False):
    """
    Computes the Pair-wise IoU (or distance) between two lists of BBs

    :param a: First List
    :param b: Second List
    :param cutoff: A cutoff - if IoU is below this, then it is converted to +/- np.Inf (depending on
            distance). Default 0 (i.e. all are valid)
    :param distance: If True, returns a distance measure (1-IoU) instead
    :return: A Matrix of size N_a by N_b
    """
    dists = np.empty([len(a), len(b)], dtype=float)
    inadm = np.PINF if distance else np.NINF
    for a_i, a_bb in enumerate(a):
        for b_i, b_bb in enumerate(b):
            iou = a_bb.iou(b_bb)
            dists[a_i, b_i] = inadm if (iou < cutoff) else (1 - iou if distance else iou)
    return dists


class BoundingBox:
    """
    A Wrapper for describing Bounding Boxes

    The class supports both accessing through properties as well as indexing style access []. Note
    that the Class deals with floating point values and as such, care must be taken when handling
    center coordinates to convert to correct pixel indices.  Note also that it assumes that image
    coordinates grow downwards and to the right, meaning that BR > TL always. No checks are done
    for this.

    Note that while I define a hash function, the class should not be taken to be immutable in the
    strict sense.
    """

    def __init__(self, tl=None, br=None, sz=None, c=None):
        """
        Initialiser.

        Two and only two of the parameters must be defined
        :param tl:  Top-Left Corner (X/Y)
        :param br:  Bottom-Right Corner (X/Y)
        :param sz:  Size (X/Y)
        :param c:   Center (X/Y)
        """
        if sum(1 for l in locals().values() if l is not None) != 3:
            raise ValueError("Exactly two of tl/br/c/sz must be specified!")

        self.__tl = np.asarray(tl, dtype=float) if tl is not None else None
        self.__br = np.asarray(br, dtype=float) if br is not None else None
        self.__c = np.asarray(c, dtype=float) if c is not None else None
        self.__sz = np.asarray(sz, dtype=float) if sz is not None else None

    def __repr__(self):
        return (
            f"[({self.top_left[0]:.1f}, {self.top_left[1]:.1f}), "
            f"({self.bottom_right[0]:.1f}, {self.bottom_right[1]:.1f})]"
        )

    def __eq__(self, other):
        """
        Equality is defined in terms of top-left and bottom-right values, and specifically,
        it is considered equal, if rounding to 1d.p. yields the same value. (In fact, it uses the
        string representation).

        Note that `other` must be a BoundingBox
        """
        return type(other) == BoundingBox and str(self) == str(other)

    def __hash__(self):
        """
        Hashing is based on the string representation of the array, and hence, is only accurate up to differences in 1 dp.
        """
        return hash(str(self))

    @property
    def top_left(self):
        if self.__tl is None:
            if self.__br is not None:
                if self.__sz is not None:
                    self.__tl = self.__br - self.__sz
                else:
                    self.__tl = self.__c - (self.__br - self.__c)
            else:
                self.__tl = self.__c - self.__sz / 2
        return self.__tl

    @property
    def bottom_right(self):
        if self.__br is None:
            if self.__tl is not None:
                if self.__sz is not None:
                    self.__br = self.__tl + self.__sz
                else:
                    self.__br = self.__c + (self.__c - self.__tl)
            else:
                self.__br = self.__c + self.__sz / 2
        return self.__br

    @property
    def size(self):
        if self.__sz is None:
            if self.__tl is not None:
                if self.__br is not None:
                    self.__sz = self.__br - self.__tl
                else:
                    self.__sz = (self.__c - self.__tl) * 2
            else:
                self.__sz = (self.__br - self.__c) * 2
        return self.__sz

    @property
    def center(self):
        if self.__c is None:
            if self.__tl is not None:
                if self.__br is not None:
                    self.__c = (self.__tl + self.__br) / 2
                else:
                    self.__c = self.__tl + self.__sz / 2
            else:
                self.__c = self.__br - self.__sz / 2
        return self.__c

    @property
    def bottom_left(self):
        return np.asarray([self.top_left[0], self.bottom_right[1]])

    @property
    def top_right(self):
        return np.asarray([self.bottom_right[0], self.top_left[1]])

    def __getitem__(self, item):
        item = item.lower()
        if item == "tl":
            return self.top_left
        elif item == "br":
            return self.bottom_right
        elif item == "c":
            return self.center
        elif item == "sz":
            return self.size
        else:
            raise ValueError("Invalid Attribute")

    def __contains__(self, item):
        """
        Returns true if the item is entirely enclosed by the bounding box

        :param item: Currently a 2D point (tuple, list or array)
        :return: True if the point item is within the Bounding Box, false otherwise
        """
        if item[0] < self.top_left[0] or item[0] > self.bottom_right[0]:
            return False
        if item[1] < self.top_left[1] or item[1] > self.bottom_right[1]:
            return False
        return True

    def area(self):
        return np.prod(self.size)

    @property
    def corners(self):
        """
        Returns all corners, in a clockwise fashion, starting from top-left
        :return:
        """
        return np.vstack(
            [
                self["tl"],  # Top-Left
                (self["br"][0], self["tl"][1]),  # Top-Right
                self["br"],  # Bottom-Right
                (self["tl"][0], self["br"][1]),  # Bottom-Left
            ]
        )

    @property
    def extrema(self):
        """
        Returns the Top-Left/Bottom-Right Extrema
        """
        return np.append(self.top_left, self.bottom_right)

    @property
    def coco(self):
        """
        Return in CoCo Format [X, Y, W, H]
        """
        return np.append(self.top_left, self.size)

    @property
    def cs(self):
        """
        Return in Center-Size format [X, Y, W, H]
        """
        return np.append(self.center, self.size)

    def iou(self, other):
        """
            Computes the Intersection over Union.
            Code modifed from: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc

            :param other:  The other bounding box instance
            :return:
            """
        # Determine the bounds of the intersection:
        x_tl = max(self.top_left[0], other.top_left[0])
        y_tl = max(self.top_left[1], other.top_left[1])
        x_br = min(self.bottom_right[0], other.bottom_right[0])
        y_br = min(self.bottom_right[1], other.bottom_right[1])

        # Compute intersection:
        intersection = max(x_br - x_tl, 0) * max(y_br - y_tl, 0)
        if intersection == 0:
            return 0

        # Compute the union
        union = self.area() + other.area() - intersection

        # Return Intersection over Union
        return intersection / union

    def overlap(self, other):
        """
        Returns the overlapping rectangle (if any)

        :param other: Another Bounding Box
        :return: Intersecting rectangle if there is any overlap, otherwise None
        """
        x_tl = max(self.top_left[0], other.top_left[0])
        y_tl = max(self.top_left[1], other.top_left[1])
        x_br = min(self.bottom_right[0], other.bottom_right[0])
        y_br = min(self.bottom_right[1], other.bottom_right[1])

        return (
            BoundingBox(tl=(x_tl, y_tl), br=(x_br, y_br)) if (x_br > x_tl and y_br > y_tl) else None
        )

    def transform(self, affine: AffineTransform, inplace=False):
        """
        Transforms the bounding box by an Affine Transform

        The method transforms the bounding box according to the specified affine transform returning
        another axis-aligned bounding box according to the following scheme:
         1. Transform the Corners of the bounding box:
         2. Find the center-point of each edge
         3. Construct an axis-aligned bounding box passing through these points

        Note that unless inplace=True, a new bounding box is returned and the object is not mutated.
        :param affine: An Affine Transform
        :param inplace: If True, update self
        :return: self or a new bounding box.
        """
        corners = affine.forward(self.corners)
        xs = np.sort(corners[:, 0]).reshape(2, 2).mean(axis=1)
        ys = np.sort(corners[:, 1]).reshape(2, 2).mean(axis=1)
        if inplace:
            self.__tl = np.asarray((xs[0], ys[0]), dtype=float)
            self.__br = np.asarray((xs[0], ys[0]), dtype=float)
            self.__c = (self.__tl + self.__br) / 2
            self.__sz = self.__br - self.__tl
            return self
        else:
            return BoundingBox(tl=(xs[0], ys[0]), br=(xs[1], ys[1]))


def build_line(pts, normalised=True):
    """
    Build a Line from two end-points
    :param pts: The end-points, either as 2x2 ([x_1, y_1], [x_2, y_2]) or 4x1 (x_1, y_1, x_2, y_2)
    :param normalised: If True, normalise to unit vector
    :return: Line representation, normalised if need be
    """
    x_1, y_1, x_2, y_2 = pts.flatten()
    if x_1 == x_2:
        if y_1 == y_2:
            raise ValueError("Line cannot be specified from two equal points.")
        _line = np.asarray((1, 0, -x_1), dtype=np.float64)
    elif y_1 == y_2:
        _line = np.asarray((0, 1, -y_1), dtype=np.float64)
    else:
        m = (y_1 - y_2) / (x_1 - x_2)
        k = y_1 - m * x_1
        _line = np.asarray((m, -1, k), dtype=np.float64)
    return _line/np.linalg.norm(_line) if normalised else _line


def line(img, pt1, pt2, color, thickness=1, lineType=8, shift=0, linestyle="-"):
    """
    Draws a line (much like cv2.line()) but allows for other line-styles.

    Args:
        linestyle: specifies the line-style: currently, supports:
            '-': Standard (default)
            '--': Dashed Line
            '.': Dotted line
        For other arguments, see cv2.line(). Note that lineType and shift are ignored if
        linestyle is not '-'
    """

    if linestyle == "-":
        return cv2.line(
            img,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color,
            thickness,
            lineType,
            shift,
        )

    elif linestyle == "--":
        pt1 = np.asarray(pt1)
        pt2 = np.asarray(pt2)

        # Some Calculations
        seg_len = 12 * thickness
        line_length = np.sqrt((np.square(pt2 - pt1)).sum())
        dvect_sld = np.divide(pt2 - pt1, line_length) * 7 * thickness
        dvect_full = np.divide(pt2 - pt1, line_length) * seg_len

        # Draw
        for i in range(np.ceil(line_length / seg_len).astype(int)):
            st = pt1 + dvect_full * i
            nd = st + dvect_sld
            cv2.line(
                img,
                (int(st[0]), int(st[1])),
                (int(min(nd[0], pt2[0])), int(min(nd[1], pt2[1]))),
                color,
                thickness,
                cv2.LINE_AA,
            )

    elif linestyle == ".":
        pt1 = np.asarray(pt1)
        pt2 = np.asarray(pt2)

        # Some Calculations
        seg_len = 4 * thickness
        line_length = np.sqrt((np.square(pt2 - pt1)).sum())
        dvect_full = np.divide(pt2 - pt1, line_length) * seg_len

        for i in range(np.ceil(line_length / seg_len).astype(int)):
            st = pt1 + dvect_full * i
            cv2.circle(img, (int(st[0]), int(st[1])), thickness, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(pt2[0]), int(pt2[1])), thickness, color, -1, cv2.LINE_AA)


def rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0, linestyle=None):
    """
    Draws a Rectangle at the specified position

    Args:
        img: OpenCV Image to draw on
        pt1: Can be either the top-left or a cvext.BoundingBox instance
        pt2: Bottom right (ignored and can be none) if pt1 is a bounding box.
        color: Color to draw with (3-tuple)
        thickness: Line Thickness. See cv.rectangle
        lineType: See cv.rectangle
        shift: See cv.rectangle
        linestyle: specifies the line-style (see cvext.line()) or the real-valued alpha parameter if
                 thickness is negative.
    """
    if type(pt1) is BoundingBox:
        pt2 = pt1["BR"]
        pt1 = pt1["TL"]

    if thickness < 0:
        alpha = float(linestyle) if linestyle is not None else 1
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color,
            -1,
            lineType,
            shift,
        )
        np.copyto(img, cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0))
    else:
        if linestyle is None or linestyle == "-":
            cv2.rectangle(
                img,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1])),
                color,
                thickness,
                lineType,
                shift,
            )
        else:
            line(img, pt1, (pt2[0], pt1[1]), color, thickness, lineType, shift, linestyle)
            line(img, (pt2[0], pt1[1]), pt2, color, thickness, lineType, shift, linestyle)
            line(img, pt1, (pt1[0], pt2[1]), color, thickness, lineType, shift, linestyle)
            line(img, (pt1[0], pt2[1]), pt2, color, thickness, lineType, shift, linestyle)


def point(img, center, color, size=1, style="."):
    """
    Draws a point at a particular x/y coordinates

    :param img: Image to modify
    :param center:  X/Y coordinates of point center
    :param color: Point Colour
    :param size: Size of the point
    :param style: Allowed styles so far are:
        * '.' Filled circle
        * 'o' Empty circle
    :return: None
    """
    if style == ".":
        cv2.circle(img, (int(center[0]), int(center[1])), size, color, -1)
    elif style == "o":
        cv2.circle(img, (int(center[0]), int(center[1])), size, color, int(math.ceil(size / 5)))
    elif style == 'x':
        ctr = (int(center[0] - 5*size), int(center[1] + 4*size))
        cv2.putText(img, 'x', ctr, cv2.FONT_HERSHEY_PLAIN, size, color, size*2)


class TimeFrame:
    """
    A Convertor class for switching between time/frame numbers.

    Note:
        time is always returned in MS.
        frames are by default 0-offset, but this can be changed in the initialiser
        floor allows one to chose between flooring or rounding when converting non-numeric values.
    """

    def __init__(self, fps=25, frm_offset=0, floor=False):
        self.FPS = fps
        self.offset = frm_offset
        self._round = np.floor if floor else np.around

    def to_frame(self, ms):
        if type(ms) == np.ndarray:
            return (self._round(ms * self.FPS / 1000) + self.offset).astype(int)
        else:
            return int(self._round(ms * self.FPS / 1000) + self.offset)

    def to_time(self, frm, ms=True):
        if ms:
            return (frm - self.offset) * 1000 / self.FPS
        else:
            return (frm - self.offset) / self.FPS


class VideoParser:
    """
    The Video Parser (Wrapper) Object

    **Note: May be deprecated soon in favour of decord**
    """

    def __init__(self, path, qsize=16):
        """
        Initialiser
        """
        self.thread = None  # Currently Nothing
        self.path = path  # Video-Capture Object
        self.queue = Queue(maxsize=qsize)  # The Queue-Size
        self.signal_stop = False  # Signal from main to thread to stop
        self.signal_started = False  # Signal from thread to main to indicate started
        self.StartAt = 0  # Where to Start
        self.Stride = 1  # Whether to stride...

        # Now some other State-Control
        self.properties = {
            VP_CUR_PROP_POS_MSEC: None,
            VP_CUR_PROP_POS_FRAMES: None,
            cv2.CAP_PROP_POS_MSEC: 0.0,
            cv2.CAP_PROP_POS_FRAMES: 0,
            cv2.CAP_PROP_FRAME_WIDTH: None,
            cv2.CAP_PROP_FRAME_HEIGHT: None,
            cv2.CAP_PROP_FPS: None,
            cv2.CAP_PROP_CONVERT_RGB: -1,
            cv2.CAP_PROP_FRAME_COUNT: -1,
            cv2.CAP_PROP_FOURCC: None,
        }

    @property
    def Size(self):
        return (
            (int(self.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if self.thread is not None
            else None
        )

    @property
    def FourCC(self):
        return int(self.get(cv2.CAP_PROP_FOURCC)) if self.thread is not None else None

    def __len__(self):
        return int(self.get(cv2.CAP_PROP_FRAME_COUNT)) if self.thread is not None else -1

    def start(self, start=None, stride=1):
        """
        Start the Parsing Loop
        :param start:   If not None (default) then signifies an index of the frame at which to
        start (0-offset)
        :param stride:  How much to stride: default is to just add 1 (this is in terms of frames).
                        Note that when striding, the last frame is ALWAYS read even if it is not
                        a multiple of the stride.
        :return:        True if successful, false otherwise
        """
        # Check that not already processing
        if self.thread is not None:
            return False

        # Open Stream, but first update signals
        self.signal_stop = False
        self.signal_started = False
        self.StartAt = start if start is not None else 0
        self.Stride = int(max(0, stride - 1))  # guard against negative striding

        # Start Thread for processing
        self.thread = Thread(target=self.__read, args=())
        self.thread.daemon = True
        self.thread.start()

        # Wait until started
        while not self.signal_started:
            tm.sleep(0.001)  # Sleep and release GIL so other thread can execute

        # Indicate success
        return True

    def read(self):
        """
        This is the read-method, with the same signature as the OpenCV one.

        Note that the method blocks if the queue is empty but there is more stuff to get. Note that
        this can be called even after calling stop, to get the remaining ones

        :return:    ret, frame
        """
        # If we have intentionally stopped, then we do not need to block and wait, since if the
        #   queue is empty, it means that nothing else will be put in there (either because the
        #   thread received the stop signal and terminated, or because the end-of-file was
        #   actually reached in the meantime!
        if self.thread is None:
            if self.queue.qsize() > 0:
                _data = self.queue.get(block=False)
                if _data[0] is not None:  # Stop signal came in after the EOF found
                    self.properties[VP_CUR_PROP_POS_MSEC] = float(
                        self.properties[cv2.CAP_PROP_POS_MSEC]
                    )
                    self.properties[cv2.CAP_PROP_POS_MSEC] = _data[0]
                    self.properties[VP_CUR_PROP_POS_FRAMES] = self.properties[
                        cv2.CAP_PROP_POS_FRAMES
                    ]
                    self.properties[cv2.CAP_PROP_POS_FRAMES] = _data[1]
                    return True, _data[2]
                return False, None
            return False, None
        # Otherwise, we need to use a while-loop to ensure that we never block indefinitely due to
        # race conditions
        else:
            _data = None
            # Get the Data, at all costs!
            while _data is None:
                try:
                    _data = self.queue.get(block=True, timeout=0.1)
                except Empty:
                    _data = None

            # Now parse
            if _data[0] is not None:
                self.properties[VP_CUR_PROP_POS_MSEC] = float(
                    self.properties[cv2.CAP_PROP_POS_MSEC]
                )
                self.properties[cv2.CAP_PROP_POS_MSEC] = _data[0]
                self.properties[VP_CUR_PROP_POS_FRAMES] = self.properties[cv2.CAP_PROP_POS_FRAMES]
                self.properties[cv2.CAP_PROP_POS_FRAMES] = _data[1]
                return True, _data[2]
            else:
                self.thread.join()
                self.thread = None
                return False, None

    def stop(self):
        """
        Stop the Parsing

        :return: None
        """
        # Nothing to stop if nothing is running
        if self.signal_stop or self.thread is None:
            return

        # Set signal to stop & join
        self.signal_stop = True
        self.thread.join()
        self.thread = None

    # Create Alias for above to be inline also with OpenCV
    release = stop

    def get(self, prop):
        """
        Get the specified property: will raise an exception if the property is not available

        :param prop: cv2 based property name
        :return:
        """
        return self.properties[prop]

    def __read(self):
        """
        Threaded method for reading from the video-capture device

        :return: None
        """
        # Start stream
        stream = cv2.VideoCapture(self.path)

        # If seeking
        if self.StartAt > 0:
            stream.set(cv2.CAP_PROP_POS_FRAMES, self.StartAt)
            assert stream.get(cv2.CAP_PROP_POS_FRAMES) == self.StartAt

        # Store/Initialise CV2s properties
        for prop in filter(lambda p: p >=0, self.properties.keys()):
            self.properties[prop] = stream.get(prop)

        # Now indicate started
        self.signal_started = True

        # Loop until stopped or end of file (through break)
        frame = []
        while not self.signal_stop and frame is not None:
            # Get the next Frame and associated Parameters
            ret, frame = stream.read()
            _fnum = stream.get(cv2.CAP_PROP_POS_FRAMES)
            # If Striding, need to add stride: however, only do this, if we are not at the end...
            if self.Stride > 0 and _fnum < self.properties[cv2.CAP_PROP_FRAME_COUNT]:
                stream.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    min(_fnum + self.Stride, self.properties[cv2.CAP_PROP_FRAME_COUNT]),
                )
                _fnum = stream.get(cv2.CAP_PROP_POS_FRAMES)
            # Check Read
            _msec = stream.get(cv2.CAP_PROP_POS_MSEC)
            # Push to queue
            while not self.signal_stop:
                try:
                    self.queue.put(
                        (_msec, _fnum, frame) if ret else (None, None, None),
                        block=True,
                        timeout=0.1,
                    )
                    break  # Break out of this inner loop
                except Full:
                    pass

        # Stop Stream
        stream.release()


class FrameGetter:
    """
    Wrapper for reading successive frames from disk, or individual frames (note that this is only a
    convenience, and is actually quite slow).

    Note that as per OpenCV defaults, the returned images are always in BGR
    """

    def __init__(self, path, fmt="{:06d}.jpg"):
        """
        Initialiser

        :param path: Path to the extracted frames or the Video File
        :param fmt:  The format mode for the name. If 'Video', then this is treated as a video
        """
        if fmt.lower() != "video" and not os.path.isdir(path):
            raise RuntimeError("A Directory must be specified when fmt is not a video.")
        self.Path = path
        self.Fmt = fmt

    def __getitem__(self, item):
        """
        Retrieve a frame by Number

        :param item: Frame Number. Note that this is translated directly to image name as per the
               fmt if reading from folder.
        :return: OpenCV Image
        :raises  AssertionError if the frame does not exist
        """
        if self.Fmt.lower() == "video":
            v = cv2.VideoCapture(self.Path)
            v.set(cv2.CAP_PROP_POS_FRAMES, item)
            assert v.get(cv2.CAP_PROP_POS_FRAMES) == item, f"Failed to Retrieve Image @ {item}"
            return v.read()[1]

        else:
            _pth = os.path.join(self.Path, self.Fmt.format(item))
            assert os.path.exists(_pth), f"Image {item} does not exist at {_pth}."
            return cv2.imread(_pth)

    def __len__(self):
        """
        Retrieves the number of Frames

        This is 'approximate', by counting the number of files matching the extension.

        :return: Length
        """
        if self.Fmt.lower() == "video":
            return int(cv2.VideoCapture(self.Path).get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return len(glob.glob(os.path.join(self.Path, f'*{os.path.splitext(self.Fmt)[1]}')))


class SwCLAHE:
    """
    Class for implementing the CLAHE algorithm by way of a sliding window. Note that this is a loose
    adaptation, and I do cut some corners in the interest of some efficiency. Note, that this
    requires the Image data to be in 8-bit Format! The reason for this implementation is to
    separate the histogram computation from the equalisation step.

    Note that the algorithm is modified as follows to allow a history:
     a) Keep track of per-pixel raw-counts in a histogram.
     b) For the true lut, perform clipping based on the number of frames used in the histogram
        computation

    Finally, while sizes follow the WxH convention, matrices are manipulated as H x W (to follow
    matrix convention of rows x column).
    """

    def __init__(self, imgSize, clipLimit=2.0, tileGridSize=(8, 8), padding="reflect"):
        """
        Initialiser

        :param imgSize:      The Dimensions of the Image (Width[C] x Height[R])
        :param clipLimit:    The Clip Limit to Employ. Note that this will be the clip-limit per
                             image added to the histogram (computed retroactively). If not
                             required, pass None
        :param tileGridSize: The Tile-Size to compute with. Note that in our case, this signifies
                             the padding around the pixel, which is a deviation from the OpenCV
                             Implementation! The padding is in terms of width and height
                             respectivel.
        :param padding:      Type of padding to employ when computing along the edges. See the
                             documentation for numpy.pad
        """
        # Store some Values for later
        self.__W, self.__H = imgSize
        self.__tile_W, self.__tile_H = tileGridSize
        tgSze = (self.__tile_W * 2 + 1) * (self.__tile_H * 2 + 1)
        self.__clip = clipLimit * tgSze / 256.0
        self.__scale = 255.0 / tgSze
        self.__pad = padding.lower()  # Ensure Lower-Case
        self.__seen = 0  # How many Images seen so far.

        # Now prepare placeholder for Histograms
        self.__hst = np.zeros([self.__H, self.__W, 256])  # Maintains Raw Counts
        self.__lut = np.zeros(
            [self.__H, self.__W, 256], dtype=np.uint8, order="C"
        )  # Maintains Clipped Counts

    def clear_histogram(self):
        """
        Clears the Histogram

        :return: self, for chaining.
        """
        # Re-Initialise Histograms
        self.__hst = np.zeros([self.__H, self.__W, 256])  # Maintains Raw Counts
        self.__lut = np.zeros(
            [self.__H, self.__W, 256], dtype=np.uint8, order="C"
        )  # Maintains Clipped Counts
        self.__seen = 0

        # Return Self
        return self

    def update_model(self, img):
        """
        Update the Histogram and perform Clipping. This creates a usable LUT, and is equivalent to
        calling update_histogram() followed by generate_lut().

        :param img: Input image to use to update the Histogram with. Must be a single channel image
                    of type uint8.
        :return:    self, for chaining
        """
        return self.update_histogram(img).generate_lut()

    def update_histogram(self, img):
        """
        Update the Histogram only, without perform any clipping

        :param img: Input image to use to update the Histogram with. Must be a single channel image
                    of type uint8
        :return:    self, for chaining
        """
        # First PAD the image: this will allow computation being much easier...
        img = np.pad(img, pad_width=[[self.__tile_H], [self.__tile_W]], mode=self.__pad)
        self.__seen += 1

        # Generate Histogram for this Image and add to the Original Histogram
        hist = np.zeros_like(self.__lut, dtype=np.uint16)
        self.__update_hist(img, self.__tile_H, self.__tile_W, hist)
        self.__hst += hist

        # Return Self
        return self

    def generate_lut(self):
        """
        Wrapper around clip function to generate LUT

        :return:  self, for chaining
        """
        # CLip
        self.__clip_limit(
            self.__hst,
            float(self.__seen * self.__clip),
            self.__lut,
            float(self.__scale / self.__seen),
        )

        # Return Self
        return self

    def transform(self, img):
        """
        Transform the Image according to the LUT.

        :param img:
        :return:
        """
        tr_ = np.empty_like(img)
        self.__transform(self.__lut, img, tr_)
        return tr_

    def apply(self, img, clear=True):
        """
        Convenience Method for joining together updating of histogram and transform... This emulates
        the default CLAHE implementation which clears the histogram after each iteration

        :param img:     The image to operate on
        :param clear:   If True, then re-generate the histogram.
        :return:        Transformed Image
        """
        if clear:
            self.clear_histogram()
        return self.update_model(img).transform(img)

    @staticmethod
    @jit(
        signature_or_function=(uint8[:, :], uint8, uint8, uint16[:, :, :]), nopython=True,
    )
    def __update_hist(padded, row_pad, col_pad, hist):
        """
        A Private Method (to just Numba) to compute the Histogram for the Padded Image

        :param padded:  The Padded Image
        :param row_pad: The padding to include along the rows (integer)
        :param col_pad: The padding to include along the columns (integer)
        :param hist:    The output histogram. This should be initialised to all zeros!
        :return:
        """
        # Compute Valid ranges for Rows and Columns
        valid_rows = (row_pad, padded.shape[0] - row_pad - 1)
        valid_cols = (
            col_pad + 1,
            padded.shape[1] - col_pad - 1,
        )  # Note that due to scheme, we start with col_pad+1

        # Now Iterate over Pixels in a Row-Column Basis
        for r_img in range(*valid_rows):
            # Get the Histogram Row we are working on...
            r_hst = r_img - row_pad
            # Compute First Pixel: Not that there is a special case when this is the top-left
            # corner, which we must compute from scratch.
            if r_hst == 0:
                for nbh_r in range(row_pad * 2 + 1):
                    for nbh_c in range(col_pad * 2 + 1):
                        hist[0, 0, padded[nbh_r, nbh_c]] += 1
            # Otherwise, we can initialise from the upper row.
            else:
                hist[r_hst, 0, :] = hist[r_hst - 1, 0, :]
                # Compute the Previous/Next Row (in image space)
                r_prev = r_img - row_pad - 1
                r_next = r_img + row_pad
                # Iterate over the columns, subtracting the previous row and adding the next one in
                # turn
                for nbh_c in range(col_pad * 2 + 1):
                    hist[r_hst, 0, padded[r_prev, nbh_c]] -= 1
                    hist[r_hst, 0, padded[r_next, nbh_c]] += 1
            # Now iterate over columns
            for c_img in range(*valid_cols):
                # Get the histogram Column we are working on: also get the previous and next columns
                c_hst = c_img - col_pad
                c_prev = c_img - col_pad - 1
                c_next = c_img + col_pad
                # Initialise the Histogram with the pixel to the left and build from there.
                hist[r_hst, c_hst, :] = hist[r_hst, c_hst - 1]
                for nbh_r in range(r_img - row_pad, r_img + row_pad + 1):
                    hist[r_hst, c_hst, padded[nbh_r, c_prev]] -= 1
                    hist[r_hst, c_hst, padded[nbh_r, c_next]] += 1

    @staticmethod
    @jit(
        signature_or_function=(double[:, :, ::1], double, uint8[:, :, ::1], double), nopython=True,
    )
    def __clip_limit(hist, limit, lut, scaler):
        """
        Here hist should be float (to avoid truncation), but lut is integer!

        :param hist:   The Original Histogram (raw)
        :param limit:  This should take into consideration the number of samples seen so far (i.e.
                       a multiplicated)
        :param lut:    The Lookup table (placeholder)
        :param scaler: should be 256 / boxsize
        :return:
        """
        # Do a Copy of Hist: This is needed to prevent unintentional modification...
        hist = hist.copy()

        # Iterate over rows/columns of Histogram
        for r in range(hist.shape[0]):
            for c in range(hist.shape[1]):
                to_clip = 0  # Initialise to_clip
                for h in range(256):
                    if hist[r, c, h] > limit:
                        to_clip += hist[r, c, h] - limit
                        hist[r, c, h] = limit
                # Now Redistribute - Note that I will ignore residuals (handled through rounding)
                hist[r, c, :] += to_clip / 256
                # Now Transform to Lookup Table (rounding) - Had to do this manually due to issues
                # with Numba!
                cumsum = 0
                for h in range(256):
                    cumsum += hist[r, c, h]
                    lut[r, c, h] = round(cumsum * scaler)

    @staticmethod
    @jit(
        signature_or_function=(uint8[:, :, ::1], uint8[:, ::1], uint8[:, ::1]), nopython=True,
    )
    def __transform(lut, img, out):
        """
        Numba JIT to transform the image (lookup table)

        :param lut: The Lookup Table
        :param img: The Original Image
        :param out: The transformed Image
        :return:
        """
        for r in range(lut.shape[0]):
            for c in range(lut.shape[1]):
                out[r, c] = lut[r, c, img[r, c]]
