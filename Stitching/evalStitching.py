import numpy as np
import matplotlib.pyplot as plt
import os
from utils import imread
from utils import showMatches
from detectBlobs import detectBlobs
from computeSift import compute_sift
from computeMatches import computeMatches
from ransac import ransac
from mergeImages import mergeImages

# Image directory
dataDir = os.path.join('..', 'data')

# Read input images
testExamples = ['book', 'hill', 'house',
                'kitchen', 'park', 'pier', 'roof', 'table']
exampleIndex = 0
imageName1 = '{}_1.jpg'.format(testExamples[exampleIndex])
imageName2 = '{}_2.jpg'.format(testExamples[exampleIndex])

im1 = imread(os.path.join(dataDir, imageName1))
im2 = imread(os.path.join(dataDir, imageName2))

# Detect keypoints
blobs1 = detectBlobs(im1, num_sigma=50, threshold=.15)
blobs2 = detectBlobs(im2, num_sigma=50, threshold=.15)

# Compute SIFT features
sift1 = compute_sift(im1, blobs1[:, 0:4])
sift2 = compute_sift(im2, blobs2[:, 0:4])

# Find the matching between features
matches = computeMatches(sift1, sift2)
showMatches(im1, im2, blobs1, blobs2, matches)

# Ransac to find correct matches and compute transformation
inliers, transf = ransac(matches, blobs1, blobs2)

goodMatches = -np.ones_like(matches)
goodMatches[inliers] = matches[inliers]

showMatches(im1, im2, blobs1, blobs2, goodMatches)

# Merge two images and display the output
stitchIm = mergeImages(im1, im2, transf)
plt.figure()
plt.imshow(stitchIm)
plt.title('stitched image: {}'.format(testExamples[exampleIndex]))
