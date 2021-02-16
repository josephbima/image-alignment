import numpy as np
import os
import errno
import matplotlib.pyplot as plt 
from skimage.feature import plot_matches


def showMatches(im1, im2, c1, c2, matches, title=""):
    disp_matches = np.array([np.arange(matches.shape[0]), matches]).T.astype(int)
    valid_matches = np.where(matches>=0)[0]
    disp_matches = disp_matches[valid_matches, :]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    #from IPython import embed; embed(); exit(-1)
    plot_matches(ax, im1, im2, 
            c1[:, [1,0]].astype(int), c2[:,[1,0]].astype(int), disp_matches)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f'../output/Q2/{title}.png')
    plt.show()


def imread(path):
    img = plt.imread(path).astype(float)
    
    #Remove alpha channel if it exists
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    #Puts images values in range [0,1]
    if img.max() > 1.0:
        img /= 255.0

    return img


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        print("Directory {} already exists.".format(dirpath))

    
#Thanks to ali_m from https://stackoverflow.com/questions/17190649
def gaussian(hsize=3,sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    shape = (hsize, hsize)
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
