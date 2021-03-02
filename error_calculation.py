import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cv2
import csv

def compare_errors(im_truth, im_compare, title='Comparison', dir=''):

    im_compare_resized = cv2.resize(im_compare, (im_truth.shape[1], im_truth.shape[0]), interpolation = cv2.INTER_AREA)

    mse = (mean_squared_error(im_truth,im_compare_resized))
    s = (ssim(im_truth,im_compare_resized, multichannel = True))

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle(f"MSE:{mse}, SSIM: {s}")

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(im_truth, cv2.COLOR_BGR2RGB))
    plt.title("Ground Truth")
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(im_compare_resized, cv2.COLOR_BGR2RGB))
    plt.title("Generated Image")
    plt.axis("off")

    # show the images
    # plt.show()
    plt.savefig(f'{dir}/error_results/{title}_figure.png')

    return mse,s


# labels=['city', 'cp', 'desert', 'field', 'forest', 'hills', 'house', 'library', 'mountain', 'seaside', 'snow']
# csv_arr = [["ground_truth","generated_img", "mse", "ssim"]]
#
# total_mse = 0
# total_ssim = 0
# i = 0
#
# for label in labels:
#     imA = cv2.imread(f'./3-dataset/{label}_truth.jpg')
#     imB = cv2.imread(f'./results_3/{label}_al.jpg')
#
#     mse,s = compare_errors(imA,imB, title=label)
#
#     total_mse += mse
#     total_ssim += s
#     i += 1
#
#     csv_arr.append([f'./3-dataset/{label}_truth.jpg',f'./results_3/{label}_al.jpg', mse, s])
#
# with open('./results_3/error_calculation.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for r in csv_arr:
#         writer.writerow(r)
#     writer.writerow(['total_average', 'total_average',total_mse/i,total_ssim/i ])
#
# print(f'Mean mse: {total_mse/i}') # 1082.118949977722
# print(f'Mean ssim: {total_ssim/i}') # 0.5135202796352917



