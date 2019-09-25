# This Module will serve as an alternative and extension to opencv - hence the name
from mpctools.extensions import npext
import numpy as np
import cv2


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