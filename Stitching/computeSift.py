import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import imread
from skimage.color import rgb2gray
from skimage.feature import corner_orientations
from skimage.feature import hog


def compute_sift(I, circles, enlarge_factor=1.5):
    """
    Args:
        I - image
        circles - Nx4 array where N is the number of circles, where the
        first column is the x-coordinate, the second column is the y-coordinate,
        the third column is the radius and the fourth is the angle of the blob
        enlarge_factor is by how much to enarge the radius of the circle before
        computing the descriptor (a factor of 1.5 or larger is usually necessary
        for best performance)
    Output:
        an Nx128 array of SIFT descriptors
    """

    sift = cv2.xfeatures2d.SIFT_create()
    angles = circles[:, 3]

    img_gray = (I.copy()*255.0).astype('uint8')

    kpts = []
    for i in range(angles.shape[0]):
        kpts.append(cv2.KeyPoint(circles[i, 0], circles[i, 1],
                                 _size=enlarge_factor*circles[i, 2],
                                 _angle=angles[i]))

    _, des = sift.compute(img_gray, kpts)
    return des


if __name__ == '__main__':
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread("../data/stitching/eg_1.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("../data/stitching/eg_2.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1 = sift.detect(img1, None)
    kp2 = sift.detect(img1, None)

    I1 = imread("../data/eg_1.jpg")
    I2 = imread("../data/eg_2.jpg")
    circles1 = [[kp1[i].pt[0], kp1[i].pt[1], kp1[i].size, kp1[i].angle]
                for i in range(len(kp1))]
    circles1 = np.array(circles1)
    circles2 = [[kp2[i].pt[0], kp2[i].pt[1], kp2[i].size, kp1[i].angle]
                for i in range(len(kp2))]
    circles2 = np.array(circles2)

    desc1 = compute_sift(I1, circles1)
    desc2 = compute_sift(I2, circles2)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(desc1, desc2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

    plt.imshow(img3), plt.show()
