"""
This module contains utilities functions used in the preprocessing phase of the detection.
"""
import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as tf
from skimage.filters import threshold_otsu
from skimage.measure import (approximate_polygon, find_contours)
from skimage.morphology import (closing, convex_hull_object, square)
from skimage.segmentation import clear_border
from scipy.spatial import ConvexHull



def normalize(imag, mean, std, debug=False):
    """
    This function returns a normalized copy of the image.
    """

    imag = imag.copy()

    # We rescale the pixels to have 0 mean and 1 standard deviation

    ##################
    # YOUR CODE HERE #
    ##################

    if debug:
        plt.imshow(imag)
        plt.title("normalized image")
        plt.show()

    return imag


def rescale(imag):
    """
    This function returns a rescaled version of the image.
    This means that values all lies in [0, 1].
    """

    imag = imag.copy()

    # We rescale the pixels to have minimum value of 0 and max value of 1.

    ##################
    # YOUR CODE HERE #
    ##################

    if debug:
        plt.imshow(imag)
        plt.title("rescaled image")
        plt.show()

    return imag

def binarize(imag, debug=False):
    """
    This function returns a binarized version of the image.
    """

    imag = imag.copy()

    # We compute an optimal threshold and form a binary image

    ##################
    # YOUR CODE HERE #
    ##################

    if debug:
        plt.imshow(imag)
        plt.title("Thresholded image")
        plt.show()

    return imag


def inverse(imag, debug=False):
    """
    This function returns an inversed image.
    """

    imag = imag.copy()

    # We inverse the an image such that former 0 values are now 1.

    ##################
    # YOUR CODE HERE #
    ##################

    if debug:
        plt.imshow(imag)
        plt.title("Thresholded image")
        plt.show()

    return imag


def approximate_square(contour):
    """
    This function approximates a contour with a square.
    """

    tol = 50

    # While the right number of segments is not found, we modify the tolerance consequently.
    for _ in range(50):
        coords = approximate_polygon(contour, tolerance=tol)
        coords = np.flip(coords[0:-1], axis=1)
        if coords.shape[0] == 4:
            return coords
        if coords.shape[0] < 4:
            print("Failed to approximate square with tolerance {}, found {} points. Retrying."\
                .format(tol, coords.shape[0]))
            tol -= 1
        else:
            print("Failed to approximate square with tolerance {}, found {} points. Retrying."\
                .format(tol, coords.shape[0]))
            tol += 1

    raise Exception("Failed to approximate square")


def reorder_contour(contour):
    """
    This function allows to reorder the contour so that the down-right point is always first, and
    points are ordered clockwise.
    """

    # We reorder the points

    ##################
    # YOUR CODE HERE #
    ##################

    return contour


def get_box_contours(imag, debug=False):
    """
    This function takes as input a single channel image, and returns a list of contours bounding
    the cubes.
    """

    # We make sure that we work on a local copy of the image
    imag = imag.copy()

    # We turn the image to a binary one

    ##################
    # YOUR CODE HERE #
    ##################

    # We extract the contours, and keep only the largest ones.

    ##################
    # YOUR CODE HERE #
    ##################

    # We approximate the contours by squares and reorder the points

    ##################
    # YOUR CODE HERE #
    ##################

    if debug:
        plt.imshow(imag)
        for coords in ctrs:
            plt.plot(coords[:, 0], coords[:, 1], 'og', linewidth=2)
            plt.plot(coords.mean(axis=0)[0], coords.mean(axis=0)[1], 'or')
            ind = [1, 2, 3, 4]
            for i, txt in enumerate(ind):
                plt.annotate(txt, (coords[i, 0], coords[i, 1]))
        plt.title("Contours found")
        plt.show()

    return ctrs


def get_sprites(imag, ctrs, debug=False):
    """
    This function computes a projective transform from the source (mnist image) to
    the destination (contour) and extracts the warped sprite.
    """

    # We make sure that we work on a local copy of the image
    imag = imag.copy()

    # We loop through the sprites 
    sprts = []

    for contour in ctrs:

        # We compute the projective transform

        ##################
        # YOUR CODE HERE #
        ##################

        # We transform the image

        ##################
        # YOUR CODE HERE #
        ##################

        if debug:
            _, axis = plt.subplots(nrows=2, figsize=(8, 3))
            axis[0].imshow(imag)
            axis[0].plot(destination_points[:, 0], destination_points[:, 1], '.r')
            axis[1].imshow(warped)
            plt.show()

        sprts.append(warped)

    return sprts


def preprocess_sprites(sprts, debug=False):
    """
    This function preprocesses sprites to make them closer to the mnist images.
    """

    out_sprites = []

    for imag in sprts:

        # We make a local copy
        imag = imag.copy()

        # We rescale, inverse and normalize

        ##################
        # YOUR CODE HERE #
        ##################

        if debug:
            plt.imshow(imag)
            plt.show()

        out_sprites.append(imag)

    return out_sprites


if __name__ == "__main__":

    import glob

    TEST = glob.glob('../data/cubes/duo/*.jpg', recursive=True)
    print(f"Found images to test: {TEST}")

    for path in TEST:
        print("Testing image {}".format(path))
        image = imageio.imread(path)[:, :, 0]
        contours = get_box_contours(image, debug=True)
        sprites = get_sprites(image, contours, debug=True)
        sprites = preprocess_sprites(sprites, debug=False)
        for img in sprites:
            plt.imshow(img)
            plt.show()
