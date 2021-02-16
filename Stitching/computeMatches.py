import numpy as np

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 4


def computeMatches(f1, f2):
    """ Match two sets of SIFT features f1 and f2 """
    def ssd(A, B):
        return ((A-B)**2).sum()

    N, d = f1.shape
    THRES = 0.8
    matches = -np.ones(N, dtype=int)
    for i, a in enumerate(f1):
        difs = []
        for j, b in enumerate(f2):
            dif = ssd(a, b)
            difs.append(dif)

        order = np.argsort(difs)
        if difs[order[0]]/difs[order[1]] < THRES:
            matches[i] = order[0]
    return matches
