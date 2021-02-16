from __future__ import print_function
import numpy as np
import cv2
import argparse

def sampleImage(img, ratio):
    sampled_image = cv2.resize(img,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=ratio,
                           fy=ratio,
                           interpolation=cv2.INTER_NEAREST)

    return sampled_image

def cropImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = img[y:y + h, x:x + w]
    return cropped

def alignImages(config):
    print(config.ref)

    im1 = cv2.imread(config.ref, cv2.IMREAD_COLOR)
    im2 = cv2.imread(config.algn, cv2.IMREAD_COLOR)

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Downsampling simulation
    # Simulate low quality image 0.25x
    im2Gray = sampleImage(im2Gray, 0.25)
    im2Gray = sampleImage(im2Gray, 4)

    cv2.imshow("graylow", im2Gray)

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


    if config.debug:

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
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    cropped_img = cropImage(im1Reg)

    cv2.imwrite(config.out, cropped_img)

    imR = cv2.resize(cropped_img, (600, 900))  # Resize image
    cv2.imshow(config.out, imR)
    cv2.waitKey(0)

    print(f'Saving image as {config.out}')
    print(f'Homography: {h}')

    return im1Reg, h


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref', type=str, default='vit2.jpg')
    parser.add_argument('--algn', type=str, default='vit1.jpg')
    parser.add_argument('--out', type=str, default='aligned.jpg')
    parser.add_argument('--matches', type=int, default=200)
    parser.add_argument('--top', type=float, default=0.15)
    parser.add_argument('--debug', type=bool, default=False)
    config = parser.parse_args()
    print(config)
    alignImages(config)

    cv2.destroyAllWindows()
