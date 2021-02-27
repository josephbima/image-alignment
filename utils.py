import cv2
import numpy as np

def getRatio(imgA,imgB):
    area_a = imgA.shape[1] * imgA.shape[0]
    print(area_a)
    area_b = imgB.shape[1] * imgB.shape[0]
    print(area_b)
    return area_a/area_b

def sampleImage(img, ratio):
    sampled_image = cv2.resize(img,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=ratio,
                           fy=ratio,
                           interpolation=cv2.INTER_NEAREST)

    return sampled_image


def removeBlackBorders(img):
    y_nonzero, x_nonzero, _ = np.nonzero(img)
    return img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

def imageCropper(img,x,xd,y,yd):
    crop_img = img[y:yd,x:xd]
    return crop_img

def splitImagesIntoThree(img, name='img'):

    img = cv2.imread(img, cv2.IMREAD_COLOR)

    # Predetermined image sizes to compute their common area
    # In the form of x,xd,y,yd
    IM1_CONFIG = (0,777,0,593)
    IM2_CONFIG = (218,1200,0,800)
    IM3_CONFIG = (0,844,232,800)

    im1 = imageCropper(img, IM1_CONFIG[0], IM1_CONFIG[1], IM1_CONFIG[2], IM1_CONFIG[3])
    im2 = imageCropper(img, IM2_CONFIG[0], IM2_CONFIG[1], IM2_CONFIG[2], IM2_CONFIG[3])
    im3 = imageCropper(img, IM3_CONFIG[0], IM3_CONFIG[1], IM3_CONFIG[2], IM3_CONFIG[3])

    # Downsample images 2 and 3
    im2 = sampleImage(im2, ratio=0.25)
    im3 = sampleImage(im3, ratio=0.5)

    # cv2.imshow('crop1', im1)
    # cv2.imshow('crop2', im2)
    # cv2.imshow('crop3', im3)
    #
    # cv2.waitKey(0)

    # Save images
    cv2.imwrite(f'{name}_1.jpg', im1)
    cv2.imwrite(f'{name}_2.jpg', im2)
    cv2.imwrite(f'{name}_3.jpg', im3)

    return

def mse(imA, imB):
    err = np.sum((imA.astype("float") - imB.astype("float")) ** 2)
    err /= float(imA.shape[0] * imA.shape[1])

    return err


# names = ['beach', 'city', 'cp', 'desert', 'field', 'forest', 'hills', 'house', 'library', 'mountain', 'seaside', 'snow']
#
# for n in names:
#     img = cv2.imread(f'./3-dataset/{n}.jpg', cv2.IMREAD_COLOR)
#     crop = imageCropper(img,218,777,232,594)
#     cv2.imwrite(f'{n}_truth.jpg', crop)
#
# im = cv2.imread('./results_3/city_al.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('ori', im)
#
# crop1 = crop(im)
# cv2.imshow('crop', crop1)
# cv2.waitKey(0)
