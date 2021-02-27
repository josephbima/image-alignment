from __future__ import print_function
import numpy as np
import cv2
import argparse
from utils import sampleImage, removeBlackBorders, getRatio


def alignImages(config):
    print(config.ref)

    im1 = cv2.imread(config.ref, cv2.IMREAD_COLOR)
    im2 = cv2.imread(config.algn, cv2.IMREAD_COLOR)

    print(im1.shape)
    print(im2.shape)

    if im1.shape[1]* im1.shape[0] > im2.shape[1]* im2.shape[0]:
        height, width, channel = im1.shape
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        height, width, channel = im2.shape
        im1 = cv2.resize(im1, (im2.shape[1], im2.shape[0]), interpolation=cv2.INTER_CUBIC)


    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Downsampling simulation
    # cv2.imshow("graylow", im2)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(config.matches)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * config.top)
    matches = matches[:numGoodMatches]


    if config.debug == True:

        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im2Gray, keypoints2, matches, None)

        im1_kp = cv2.drawKeypoints(im1Gray, keypoints1, None, flags=None)
        im2_kp = cv2.drawKeypoints(im2Gray, keypoints2, None, flags=None)

        imS1 = cv2.resize(im1_kp, (600, 900))  # Resize image
        imS2 = cv2.resize(im2_kp, (600, 900))  # Resize image

        cv2.imshow("im3", imS1)
        cv2.imshow("im4", imS2)

        imM = cv2.resize(imMatches, (600, 900))  # Resize image

        cv2.imshow("matches", imM)


    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    im1Reg = cv2.warpPerspective(im1, h, (1200, 900))

    cv2.imshow('with borders', im1Reg)

    cropped_img = removeBlackBorders(im1Reg)

    cv2.imwrite(config.out, cropped_img)

    cv2.imshow(config.out, cropped_img)

    cv2.waitKey(0)

    print(f'Saving image as {config.out}')
    print(f'Homography: {h}')

    print(cropped_img.shape[0])
    print(cropped_img.shape[1])
    return im1Reg, h


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--ref', type=str, default='vit2.jpg')
    parser.add_argument('--algn', type=str, default='vit1.jpg')
    parser.add_argument('--out', type=str, default='aligned.jpg')
    parser.add_argument('--matches', type=int, default=200)
    parser.add_argument('--top', type=float, default=0.15)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--ratio', type=int, default=1)

    config = parser.parse_args()
    print(config)
    alignImages(config)

    cv2.destroyAllWindows()
