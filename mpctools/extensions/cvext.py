# This Module will serve as an alternative and extension to opencv - hence the name
from mpctools.extensions import npext
from queue import Queue, Empty, Full
from threading import Thread
import numpy as np
import time as tm
import cv2


# Define Some Constants
VP_CUR_PROP_POS_MSEC = -100
VP_CUR_PROP_POS_FRAMES = -101


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
        return np.squeeze(cv2.perspectiveTransform(np.expand_dims(npext.ensure2d(points, axis=0), axis=0), self.toImg))

    def to_world(self, points):
        """
        Convert Image coordinates to world-coordinates

        :param points: 2D Numpy array, with the last dimension of size 2 (X/Y coordinates): if 1D will be automatically
                       promoted to 2D
        :return:    World Coordinates
        """
        return np.squeeze(cv2.perspectiveTransform(np.expand_dims(npext.ensure2d(points, axis=0), axis=0), self.toWrld))


def expand_box(c, size):
    """
    Create a Rectangle of the specified size, centred at the point. Note that since this is aimed for images, it assumes
    that Y grows downwards (this is relevant in specifying what is meant by the top-left corner)

    :param c:       Centre (2-tuple/array, X/Y)
    :param size:    The size of the rectangle (2-tuple/array, width/height)
    :return:        Four corners of the bounding box, clockwise, from top-left corner
    """
    x, y = np.asarray(size)/2
    return np.asarray(((c - (x, y)), (c + (x, -y)), (c + (x, y)), (c + (-x, y))))


class VideoParser:
    """
    The Video Parser (Wrapper) Object
    """
    def __init__(self, path, qsize=16):
        """
        Initialiser
        """
        self.thread = None                  # Currently Nothing
        self.path = path                    # Video-Capture Object
        self.queue = Queue(maxsize=qsize)   # The Queue-Size
        self.signal_stop = False            # Signal from main to thread to stop
        self.signal_started = False         # Signal from thread to main to indicate started
        self.StartAt = 0                    # Where to Start

        # Now some other State-Control
        self.properties = {VP_CUR_PROP_POS_MSEC: None,
                           VP_CUR_PROP_POS_FRAMES: None,
                           cv2.CAP_PROP_POS_MSEC: 0.0,
                           cv2.CAP_PROP_POS_FRAMES: 0,
                           cv2.CAP_PROP_FRAME_WIDTH: None,
                           cv2.CAP_PROP_FRAME_HEIGHT: None,
                           cv2.CAP_PROP_FPS: None,
                           cv2.CAP_PROP_CONVERT_RGB: -1,
                           cv2.CAP_PROP_FRAME_COUNT: -1,
                           cv2.CAP_PROP_FOURCC: None}

    def start(self, start=None):
        """
        Start the Parsing Loop
        :param start:   If not None (default) then signifies an index of the frame at which to start
        :return:        True if successful, false otherwise
        """
        # Check that not already processing
        if self.thread is not None:
            return False

        # Open Stream, but first update signals
        self.signal_stop = False
        self.signal_started = False
        self.StartAt = start if start is not None else 0

        # Start Thread for processing
        self.thread = Thread(target=self.__read, args=())
        self.thread.daemon = True
        self.thread.start()

        # Wait until started
        while not self.signal_started:
            tm.sleep(0.001) # Sleep and release GIL so other thread can execute

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
                if _data[0] is not None:  # Because it could happen that the stop signal came in after the EOF found
                    self.properties[VP_CUR_PROP_POS_MSEC] = float(self.properties[cv2.CAP_PROP_POS_MSEC])
                    self.properties[cv2.CAP_PROP_POS_MSEC] = _data[0]
                    self.properties[VP_CUR_PROP_POS_FRAMES] = self.properties[cv2.CAP_PROP_POS_FRAMES]
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
                self.properties[VP_CUR_PROP_POS_MSEC] = float(self.properties[cv2.CAP_PROP_POS_MSEC])
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
        if self.signal_stop or self.thread is None: return

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
            assert(stream.get(cv2.CAP_PROP_POS_FRAMES) == self.StartAt)

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
            _msec = stream.get(cv2.CAP_PROP_POS_MSEC)
            _fnum = stream.get(cv2.CAP_PROP_POS_FRAMES)
            # Push to queue
            while not self.signal_stop:
                try:
                    self.queue.put((_msec, _fnum, frame) if ret else (None, None, None), block=True, timeout=0.1)
                    break   # Break out of this inner loop
                except Full:
                    pass

        # Stop Stream
        stream.release()