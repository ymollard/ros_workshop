"""
SThis module contains utilities functions used in the preprocessing phase of the detection.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, convex_hull_object, disk, white_tophat
from skimage import exposure
from skimage.color import label2rgb
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
import imageio
import copy
import numpy as np
import skimage.transform as tf

def normalize(image, mean, std, debug=False):
    """
    This function returns a normalized copy of the image.
    """

    image = image.copy()

    # We rescale the pixels to have zero mean and 1 standard deviation
    image = image - mean
    image = image / std

    if debug:
        plt.imshow(image)
        plt.title("normalized image")
        plt.show()

    return image


def rescale(image, debug=False):
    """
    This function returns a rescaled copy of the image.
    """

    return normalize(image, image.min(), image.max())


def binarize(image, debug=False):
    """
    This function returns a binarized image of the image.
    """

    image = image.copy()

    # We compute an optimal threshold and form a binary image
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    cleared = clear_border(bw)
    cleared = convex_hull_object(cleared)

    if debug:
        plt.imshow(cleared)
        plt.title("Thresholded image")
        plt.show()

    return cleared

def inverse(image, debug=False):
    """
    This function returns an inversed image of a normalized image.
    """

    image = image.copy()

    image = 1. - image

    if debug:
        plt.imshow(image)
        plt.title("Thresholded image")
        plt.show()

    return image

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
        elif coords.shape[0] < 4:
            print("Failed to approximate square with tolerance {}, found {} points. Retrying."\
                .format(tol, coords.shape[0]))
            tol -= 5
        else:
            print("Failed to approximate square with tolerance {}, found {} points. Retrying."\
                .format(tol, coords.shape[0]))
            tol += 5
    raise Exception("Failed to approximate square")


def get_box_contours(image, debug=False):
    """
    This function takes as input a single channel image, and returns a list of contours bounding 
    the cubes.
    """

    # We make sure that we work on a local copy of the image
    image = image.copy()

    # We preprocess the images
    image = binarize(rescale(image, debug=debug), debug=debug)


    # We find the contours
    contours = [approximate_square(c) for c in find_contours(image, 0)]
    if debug:
        plt.imshow(image)
        for coords in contours:
            plt.plot(coords[:, 0], coords[:, 1], 'og', linewidth=2)
            plt.plot(coords.mean(axis=0)[0], coords.mean(axis=0)[1],  'or')
        plt.title("Contours found")
        plt.show()

    return contours


def get_sprites(image, contours, debug=False):
    """
    For each of the given contours, this function computes the transform and extracts the warped 
    sprite.
    """

    # We make sure that we work on a local copy of the image
    image = image.copy()

    sprites = []
    for contour in contours:

        # We retrieve the coordinates of the source and the destination points
        source_points = np.array([[28, 28], [0, 28], [0, 0], [28, 0]])
        destination_points = contour

        # We compute the transform
        transform = tf.ProjectiveTransform()
        transform.estimate(source_points, destination_points)
        warped = tf.warp(image, transform, output_shape=(28, 28))
        if debug:
            _, ax = plt.subplots(nrows=2, figsize=(8, 3))
            ax[0].imshow(image)
            ax[0].plot(destination_points[:, 0], destination_points[:, 1], '.r')
            ax[1].imshow(warped)
            plt.show()

        sprites.append(warped)

    return sprites


def preprocess_sprites(sprites, debug=False):
    """
    This function preprocesses sprites to make them closer to the mnist images.
    """

    out_sprites = []

    for img in sprites:

        # We make a local copy
        img = img.copy()

        # We rescale, inverse and normalize
        img = inverse(rescale(img, debug=debug), debug=debug)
        img = normalize(img, 0.5, 0.5, debug=debug)

        if debug:
            plt.imshow(img)
            plt.show()

        out_sprites.append(img)
    
    return out_sprites


if __name__=="__main__":
    
    import glob

    test = glob.glob('data/cubes/**/*.jpg', recursive=True)

    for path in test:
        print("Testing image {}".format(path))
        image = imageio.imread(path)[:,:,0]
        contours = get_box_contours(image, debug=False)
        sprites = get_sprites(image, contours, debug=False)
        sprites = preprocess_sprites(sprites, debug=False)
        for img in sprites:
            plt.imshow(img)
            plt.show()