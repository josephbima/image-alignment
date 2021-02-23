import cv2

def sampleImage(img, ratio):
    sampled_image = cv2.resize(img,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=ratio,
                           fy=ratio,
                           interpolation=cv2.INTER_NEAREST)

    return sampled_image

def removeBlackBorders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    print(x,y,w,h)

    cropped = img[y:y + h, x:x + w]
    return cropped

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

# names = ['beach', 'city', 'cp', 'desert', 'field', 'forest', 'hills', 'house', 'library', 'mountain', 'seaside', 'snow']
#
# for n in names:
#     img = cv2.imread(f'./3-dataset/{n}.jpg', cv2.IMREAD_COLOR)
#     crop = imageCropper(img,218,777,232,594)
#     cv2.imwrite(f'{n}_truth.jpg', crop)