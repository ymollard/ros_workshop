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

    # We rescale the pixels to have zero mean and 1 standard deviation
    imag = imag - mean
    imag = imag / std

    if debug:
        plt.imshow(imag)
        plt.title("normalized image")
        plt.show()

    return imag


def rescale(imag):
    """
    This function returns a rescaled copy of the image.
    """

    return normalize(imag, imag.min(), imag.max())


def binarize(imag, debug=False):
    """
    This function returns a binarized image of the image.
    """

    imag = imag.copy()

    # We compute an optimal threshold and form a binary image
    thresh = threshold_otsu(imag)
    black_white = closing(imag > thresh, square(3))
    cleared = clear_border(black_white)
    cleared = convex_hull_object(cleared)

    if debug:
        plt.imshow(cleared)
        plt.title("Thresholded image")
        plt.show()

    return cleared


def inverse(imag, debug=False):
    """
    This function returns an inversed image of a normalized image.
    """

    imag = imag.copy()

    imag = 1. - imag

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
    The contour found by `approximate_square` may not be always sorted in the same way. For instance
    we want the down-right point to always be one. This function allows to reorder the contour this
    way.
    """

    rightest_points = contour[contour[:, 0].argsort()][-2:]
    lowest_rightest_point = rightest_points[rightest_points[:, 1].argsort()][-1]
    lr_point_idx = np.where(np.all(contour == lowest_rightest_point, axis=1))[0][0]
    return np.roll(contour, -lr_point_idx, axis=0)


def get_box_contours(imag, debug=False):
    """
    This function takes as input a single channel image, and returns a list of contours bounding
    the cubes.
    """

    # We make sure that we work on a local copy of the image
    imag = imag.copy()

    # We preprocess the images
    imag = binarize(rescale(imag), debug=debug)

    # We extract the convex hulls, and keep only the largest ones.
    ctrs = find_contours(imag, 0)
    hulls = [ConvexHull(c) for c in ctrs]
    ctrs = [h.points[np.flip(h.vertices)] for h in hulls if h.area > 500]

    # We keep the convex hull of the cont
    ctrs = [reorder_contour(approximate_square(c)) for c in ctrs]
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
    For each of the given contours, this function computes the transform and extracts the warped
    sprite.
    """

    # We make sure that we work on a local copy of the image
    imag = imag.copy()

    sprts = []
    for contour in ctrs:

        # We retrieve the coordinates of the source and the destination points
        source_points = np.array([[28, 28], [0, 28], [0, 0], [28, 0]])
        destination_points = contour

        # We compute the transform
        transform = tf.ProjectiveTransform()
        transform.estimate(source_points, destination_points)
        warped = tf.warp(imag, transform, output_shape=(28, 28))
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
        imag = inverse(rescale(imag), debug=debug)
        imag = normalize(imag, 0.5, 0.5, debug=debug)

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
