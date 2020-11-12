"""
This Module will serve as an alternative and extension to opencv - hence the name

Some notes re compatibility with OpenCV.

 1. Sizes: OpenCV expects sizes to be passed as [W x H]. However, in following with normal matrix
    manipulations, Images are passed as [H x W x 3]

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

from numba import jit, uint8, uint16, double
from mpctools.extensions import npext, utils
from queue import Queue, Empty, Full
from threading import Thread
import numpy as np
import time as tm
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
        Based on code by Dr Rowland Sillito, ActualAnalytics
    """

    def __init__(self, image_coords, world_coords):
        """
        Initialiser

        :param image_coords: A 2D Numpy array of image coordinates, with x/y along the second axis.
                             Must be of length at least 4.
        :param world_coords: A 2D Numpy arra of corresponding world-coordinates: must be same shape
                            as image_coords
        """
        self.toImg = cv2.findHomography(world_coords, image_coords)[0]
        self.toWrld = cv2.findHomography(image_coords, world_coords)[0]

    def to_image(self, points):
        """
        Convert world coordinates to image coordinates

        :param points: 2D Numpy array, with the last dimension of size 2 (X/Y coordinates): if 1D
                       will be automatically promoted to 2D
        :return:    Image Coordinates
        """
        points2d = npext.ensure2d(points, axis=0)
        valid = np.isfinite(points2d).all(axis=1)
        img_coords = np.full_like(points2d, fill_value=np.NaN)
        img_coords[valid, :] = np.squeeze(
            cv2.perspectiveTransform(np.expand_dims(points2d[valid, :], axis=0), self.toImg)
        )
        return img_coords

    def to_world(self, points):
        """
        Convert Image coordinates to world-coordinates

        :param points: 2D Numpy array, with the last dimension of size 2 (X/Y coordinates): if 1D
                       will be automatically promoted to 2D
        :return:    World Coordinates
        """
        points2d = npext.ensure2d(points, axis=0)
        valid = np.isfinite(points2d).all(axis=1)
        wd_coords = np.full_like(points2d, fill_value=np.NaN)
        wd_coords[valid, :] = np.squeeze(
            cv2.perspectiveTransform(np.expand_dims(points2d[valid, :], axis=0), self.toWrld)
        )


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
    for a_i, a_bb in enumerate(a):
        for b_i, b_bb in enumerate(b):
            iou = a_bb.iou(b_bb)
            dists[a_i, b_i] = (np.PINF if distance else np.NINF) if (iou < cutoff) else (1 - iou if distance else iou)
    return dists


class BoundingBox:
    """
    A Wrapper for describing Bounding Boxes

    The class supports both accessing through properties as well as indexing style access []. Note
    that the Class deals with floating point values and as such, care must be taken when handling
    center coordinates to convert to correct pixel indices.  Note also that it assumes that image
    coordinates grow downwards and to the right, meaning that BR > TL always. No checks are done
    for this.

    In addition, the class is generally immutable, and hence can be hashable.
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

        self.TL = np.asarray(tl, dtype=float) if tl is not None else None
        self.BR = np.asarray(br, dtype=float) if br is not None else None
        self.C = np.asarray(c, dtype=float) if c is not None else None
        self.SZ = np.asarray(sz, dtype=float) if sz is not None else None

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
        Since we do not expect to mutate the object, we can hash it. Hashing is based on the
        string representation of the array, and hence, is only accurate up to differences in 1 dp.
        """
        return hash(str(self))

    @property
    def top_left(self):
        if self.TL is None:
            if self.BR is not None:
                if self.SZ is not None:
                    self.TL = self.BR - self.SZ
                else:
                    self.TL = self.C - (self.BR - self.C)
            else:
                self.TL = self.C - self.SZ / 2
        return self.TL

    @property
    def bottom_right(self):
        if self.BR is None:
            if self.TL is not None:
                if self.SZ is not None:
                    self.BR = self.TL + self.SZ
                else:
                    self.BR = self.C + (self.C - self.TL)
            else:
                self.BR = self.C + self.SZ / 2
        return self.BR

    @property
    def size(self):
        if self.SZ is None:
            if self.TL is not None:
                if self.BR is not None:
                    self.SZ = self.BR - self.TL
                else:
                    self.SZ = (self.C - self.TL) * 2
            else:
                self.SZ = (self.BR - self.C) * 2
        return self.SZ

    @property
    def center(self):
        if self.C is None:
            if self.TL is not None:
                if self.BR is not None:
                    self.C = (self.TL + self.BR) / 2
                else:
                    self.C = self.TL + self.SZ / 2
            else:
                self.C = self.BR - self.SZ / 2
        return self.C

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
        x, y = self.size / 2
        c = self.center
        return np.asarray(((c - (x, y)), (c + (x, -y)), (c + (x, y)), (c + (-x, y))))

    @property
    def extrema(self):
        """
        Returns the Top-Left/Bottom-Right Extrema
        """
        return np.append(self.top_left, self.bottom_right)

    @property
    def mot_format(self):
        return np.append(self.top_left, self.size)

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

    @TODO: Some fancy checks to ensure that line ends with a dash.
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
        cv2.rectangle(overlay, (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color,
            -1,
            lineType,
            shift,)
        np.copyto(img, cv2.addWeighted(overlay, alpha, img, 1-alpha, 0))
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


class TimeFrame:
    """
    A Convertor class for switching between time/frame numbers.

    Note:
        time is always returned in MS.
        frames are by default 0-offset, but this can be changed in the initialiser
    """

    def __init__(self, fps=25, frm_offset=0):
        self.FPS = fps
        self.offset = frm_offset

    def to_frame(self, ms):
        if type(ms) == np.ndarray:
            return (np.around(ms * self.FPS / 1000) + self.offset).astype(int)
        else:
            return int(np.round(ms * self.FPS / 1000) + self.offset)

    def to_time(self, frm):
        return (frm - self.offset) * 1000 / self.FPS


class VideoParser:
    """
    The Video Parser (Wrapper) Object
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

        # Store/Initialise some properties
        self.properties[cv2.CAP_PROP_POS_MSEC] = stream.get(cv2.CAP_PROP_POS_MSEC)
        self.properties[cv2.CAP_PROP_POS_FRAMES] = stream.get(cv2.CAP_PROP_POS_FRAMES)
        self.properties[cv2.CAP_PROP_FRAME_HEIGHT] = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.properties[cv2.CAP_PROP_FRAME_WIDTH] = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.properties[cv2.CAP_PROP_FPS] = stream.get(cv2.CAP_PROP_FPS)
        self.properties[cv2.CAP_PROP_FRAME_COUNT] = stream.get(cv2.CAP_PROP_FRAME_COUNT)
        self.properties[cv2.CAP_PROP_FOURCC] = stream.get(cv2.CAP_PROP_FOURCC)
        self.properties[cv2.CAP_PROP_CONVERT_RGB] = stream.get(cv2.CAP_PROP_CONVERT_RGB)

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
    Wrapper for reading successive frames from disk.

    Note that as per OpenCV defaults, the returned images are always in BGR
    """

    def __init__(self, path, fmt="{:06d}.jpg"):
        """
        Initialiser

        :param path: Path to the extracted frames
        :param fmt:  The format mode for the name
        """
        if not os.path.isdir(path):
            raise RuntimeError(f'Specified Directory "{path}" does not exist!')
        self.Path = path
        self.Fmt = fmt

    def __getitem__(self, item):
        """
        Retrieve a frame by Number

        :param item: Frame Number. Note that this is translated directly to image name.
        :return: OpenCV Image
        :raises  ValueError if the frame does not exist
        """
        _pth = os.path.join(self.Path, self.Fmt.format(item))
        if os.path.exists(_pth):
            return cv2.imread(_pth)
        raise ValueError(f"Image {item} does not exist.")


class SwCLAHE:
    """
    Class for implementing the CLAHE algorithm by way of a sliding window. Note that this is a loose
    adaptation, and I do cut some corners in the interest of some efficiency. Note, that this
    requires the Image data to be in 8-bit Format! The reason for this implementation is to
    separate the histogram computation from the equalisation step.

    Note that the algorithm is modified as follows to allow a history:
     a) Keep track of per-pixel raw-counts in a histogram.
     b) For the true lut, perform clipping based on the number of frames used in the histogram
        computation.
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
