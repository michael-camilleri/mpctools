"""
This Module will serve as an alternative and extension to opencv - hence the name

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""

from numba import jit, uint8, uint16, double
from mpctools.extensions import npext
from queue import Queue, Empty, Full
from threading import Thread
import numpy as np
import time as tm
import cv2


# Define Some Constants
VP_CUR_PROP_POS_MSEC = -100  # Current frame (rather than next one) in MS
VP_CUR_PROP_POS_FRAMES = -101  # Current frame (rather than next one) in index


class Homography:
    """
    Class for fitting a Homography:
        Based on code by Dr Rowland Sillito, ActualAnalytics
    """

    def __init__(self, image_coords, world_coords):
        """
        Initialiser

        :param image_coords: A 2D Numpy array of image coordinates, with x/y along the second axis. Must be of length at
                                least 4.
        :param world_coords: A 2D Numpy arra of corresponding world-coordinates: must be same shape as image_coords
        """
        self.toImg = cv2.findHomography(world_coords, image_coords)[0]
        self.toWrld = cv2.findHomography(image_coords, world_coords)[0]

    def to_image(self, points):
        """
        Convert world coordinates to image coordinates

        :param points: 2D Numpy array, with the last dimension of size 2 (X/Y coordinates): if 1D will be automatically
                       promoted to 2D
        :return:    Image Coordinates
        """
        return np.squeeze(
            cv2.perspectiveTransform(
                np.expand_dims(npext.ensure2d(points, axis=0), axis=0), self.toImg
            )
        )

    def to_world(self, points):
        """
        Convert Image coordinates to world-coordinates

        :param points: 2D Numpy array, with the last dimension of size 2 (X/Y coordinates): if 1D will be automatically
                       promoted to 2D
        :return:    World Coordinates
        """
        return np.squeeze(
            cv2.perspectiveTransform(
                np.expand_dims(npext.ensure2d(points, axis=0), axis=0), self.toWrld
            )
        )


def expand_box(c, size):
    """
    Create a Rectangle of the specified size, centred at the point. Note that since this is aimed for images, it assumes
    that Y grows downwards (this is relevant in specifying what is meant by the top-left corner)

    :param c:       Centre (2-tuple/array, X/Y)
    :param size:    The size of the rectangle (2-tuple/array, width/height)
    :return:        Four corners of the bounding box, clockwise, from top-left corner
    """
    x, y = np.asarray(size) / 2
    return np.asarray(((c - (x, y)), (c + (x, -y)), (c + (x, y)), (c + (-x, y))))


def intersection_over_union(ground_truth, prediction):
    """
    Computes the Intersection over Union.
    Code modifed from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc

    :param ground_truth: The ground-truth rectangle. Stored as [X_tl, Y_tl, W, H]
    :param prediction:   The predicted rectangle. Stored as [X_tl, Y_tl, W, H]
    :return:
    """
    # Determine the bounds of the intersection:
    x_tl = max(ground_truth[0], prediction[0])
    y_tl = max(ground_truth[1], prediction[1])
    x_br = min(ground_truth[0] + ground_truth[2], prediction[0] + prediction[2])
    y_br = min(ground_truth[1] + ground_truth[3], prediction[1] + prediction[3])

    # Compute intersection:
    intersection = max(x_br - x_tl, 0) * max(y_br - y_tl, 0)
    if intersection == 0:
        return 0

    # Compute the union
    union = (
        ground_truth[2] * ground_truth[3] + prediction[2] * prediction[3] - intersection
    )

    # Return Intersection over Union
    return intersection / union


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

    def start(self, start=None, stride=1):
        """
        Start the Parsing Loop
        :param start:   If not None (default) then signifies an index of the frame at which to start
        :param stride:  How much to stride: default is to just add 1... (this is in terms of frames). Note that when
                        striding, the last frame is ALWAYS read even if it is not a multiple of the stride...
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

        Note that the method blocks if the queue is empty but there is more stuff to get. Note that this can be called
        even after calling stop, to get the remaining ones

        :return:    ret, frame
        """
        # If we have intentionally stopped, then we do not need to block and wait, since if the queue is empty, it
        #   means that nothing else will be put in there (either because the thread received the stop signal and
        #   terminated, or because the end-of-file was actually reached in the meantime!
        if self.thread is None:
            if self.queue.qsize() > 0:
                _data = self.queue.get(block=False)
                if (
                    _data[0] is not None
                ):  # Because it could happen that the stop signal came in after the EOF found
                    self.properties[VP_CUR_PROP_POS_MSEC] = float(
                        self.properties[cv2.CAP_PROP_POS_MSEC]
                    )
                    self.properties[cv2.CAP_PROP_POS_MSEC] = _data[0]
                    self.properties[VP_CUR_PROP_POS_FRAMES] = self.properties[
                        cv2.CAP_PROP_POS_FRAMES
                    ]
                    self.properties[cv2.CAP_PROP_POS_FRAMES] = _data[1]
                    return True, _data[2]
                else:
                    return False, None
            else:
                return False, None
        # Otherwise, we need to use a while-loop to ensure that we never block indefinitely due to race conditions
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
                self.properties[VP_CUR_PROP_POS_FRAMES] = self.properties[
                    cv2.CAP_PROP_POS_FRAMES
                ]
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
        self.properties[cv2.CAP_PROP_FRAME_HEIGHT] = stream.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
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


class SwCLAHE:
    """
    Class for implementing the CLAHE algorithm by way of a sliding window. Note that this is a loose adaptation, and I
    do cut some corners in the interest of some efficiency. Note, that this requires the Image data to be in 8-bit
    Format! The reason for this implementation is to separate the histogram computation from the equalisation step.

    Note that the algorithm is modified as follows to allow a history:
     a) Keep track of per-pixel raw-counts in a histogram.
     b) For the true lut, perform clipping based on the number of frames used in the histogram computation.
    """

    def __init__(self, imgSize, clipLimit=2.0, tileGridSize=(8, 8), padding="reflect"):
        """
        Initialiser

        :param imgSize:         The Dimensions of the Image (Width[C] x Height[R])
        :param clipLimit:       The Clip Limit to Employ. Note that this will be the clip-limit per image added to the
                                histogram (computed retroactively). If not required, pass None
        :param tileGridSize:    The Tile-Size to compute with. Note that in our case, this signifies the padding around
                                the pixel, which is a deviation from the OpenCV Implementation! The padding is in terms
                                of width and height respectivel.
        :param padding:         Type of padding to employ when computing along the edges. See the documentation for
                                numpy.pad
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
        Update the Histogram and perform Clipping. This creates a usable LUT, and is equivalent to calling
        update_histogram() followed by generate_lut().

        :param img: Input image to use to update the Histogram with. Must be a single channel image of type uint8.
        :return:    self, for chaining
        """
        return self.update_histogram(img).generate_lut()

    def update_histogram(self, img):
        """
        Update the Histogram only, without perform any clipping

        :param img: Input image to use to update the Histogram with. Must be a single channel image of type uint8
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
        Convenience Method for joining together updating of histogram and transform... This emulates the default
        CLAHE implementation which clears the histogram after each iteration

        :param img:     The image to operate on
        :param clear:   If True, then re-generate the histogram.
        :return:        Transformed Image
        """
        if clear:
            self.clear_histogram()
        return self.update_model(img).transform(img)

    @staticmethod
    @jit(
        signature_or_function=(uint8[:, :], uint8, uint8, uint16[:, :, :]),
        nopython=True,
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
            # Compute First Pixel:
            # Not that there is a special case when this is the top-left corner, which we must compute from scratch.
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
                # Iterate over the columns, subtracting the previous row and adding the next one in turn
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
        signature_or_function=(double[:, :, ::1], double, uint8[:, :, ::1], double),
        nopython=True,
    )
    def __clip_limit(hist, limit, lut, scaler):
        """
        Here hist should be float (to avoid truncation), but lut is integer!

        :param hist:   The Original Histogram (raw)
        :param limit:  This should take into consideration the number of samples seen so far (i.e. a multiplicated)
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
                # Now Transform to Lookup Table (rounding) - Had to do this manually due to issues with Numba!
                cumsum = 0
                for h in range(256):
                    cumsum += hist[r, c, h]
                    lut[r, c, h] = round(cumsum * scaler)

    @staticmethod
    @jit(
        signature_or_function=(uint8[:, :, ::1], uint8[:, ::1], uint8[:, ::1]),
        nopython=True,
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
