import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import rescale
from skimage.transform import SimilarityTransform
from skimage.transform import warp

from utils import imread

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2018
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 3

def mergeImages(im1, im2, transf):

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    #Coordinates of the four corners
    c1 = np.array([[0, w1, w1, 0],[0, 0, h1, h1]]).astype(int)
    c2 = np.array([[0, w2, w2, 0],[0, 0, h2, h2]]).astype(int)

    #Transformed coordinates
    fulltransf = np.eye(3)
    fulltransf[0:2, :] = transf

    a = np.ones((1, c2.shape[1]))
    v = np.concatenate((c2, 
        np.ones((1, c2.shape[1]))),
        axis=0)
    tc2 = fulltransf.dot(v)[0:2, :]
    all_corners = np.concatenate((c1, tc2), axis=1)

    corner_min = np.min(all_corners, axis=1)
    corner_max = np.max(all_corners, axis=1)
    output_shape = corner_max - corner_min
    output_shape = np.ceil(output_shape[::-1]).astype(int)

    offset1 = SimilarityTransform(translation=-corner_min)
    t2 = SimilarityTransform(fulltransf)

    mask1 = np.ones_like(im1)
    mask2 = np.ones_like(im2)
    im1warp = warp(im1, offset1.inverse, output_shape=output_shape)
    im2warp = warp(im2, (offset1 + t2).inverse, output_shape=output_shape)

    mask1warp = warp(mask1, offset1.inverse, output_shape=output_shape)
    mask2warp = warp(mask1, (offset1 + t2).inverse, output_shape=output_shape)
    overlap = np.logical_and(mask1warp, mask2warp)

    merged = im1warp + im2warp
    merged[overlap] = im2warp[overlap]
    return merged


if __name__ == '__main__':
    im1 = imread("../data/eg_1.jpg")
    im2 = imread("../data/eg_2.jpg")
    transf = np.array([[1.0047, 0.0445, 141.4735],
            [-0.0409, 1.0012, -17.7628]])
    stitched = mergeImages(im1, im2, transf)
    plt.imshow(stitched)
    plt.show()
