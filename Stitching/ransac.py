from math import sqrt
import numpy as np

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 4


def ransac(matches, blobs1, blobs2):
    N = 500
    THRES = 10
    valid = np.where(matches != -1)[0]
    best_m = np.empty((2, 3), dtype=float)

    def calculate_matrix():  # matches, valid, blobs1, blobs2):
        seeds = np.random.choice(valid, 3)
        pairs = [(i, j) for i, j in zip(seeds, matches[seeds])]
        left, right1, right2 = [], [], []
        for pair in pairs:
            idx1, idx2 = pair
            left.append(np.append(blobs2[idx2][:2], 1))
            right1.append(blobs1[idx1][0:1])
            right2.append(blobs1[idx1][1:2])
        (m1, m2, t1) = np.linalg.solve(left, right1)
        (m3, m4, t2) = np.linalg.solve(left, right2)
        M = np.array([
            [m1[0], m2[0], t1[0]],
            [m3[0], m4[0], t2[0]]
        ])
        return M

    def calculate_error(M, source, target):
        return sqrt(((M[0] * np.append(source[:2], 1)).sum() - target[0])**2 + ((M[1] * np.append(source[:2], 1)).sum() - target[1])**2)

    def total_err(M):
        inliers = []
        for idx in valid:
            source = blobs2[matches[idx], :2]
            target = blobs1[idx, :2]
            single_err = calculate_error(M, source, target)
            if single_err < THRES:
                inliers.append(idx)
        return inliers

    best_inliers_num = 0
    best_inliers = None
    best_m = None

    for _ in range(N):
        try:
            M = calculate_matrix()
        except np.linalg.LinAlgError:
            continue
        inliers = total_err(M)
        # print('Total ERROR: ', err, '\n')
        if len(inliers) > best_inliers_num:
            best_inliers_num = len(inliers)
            best_inliers = inliers
            best_m = M
    return best_inliers, best_m
