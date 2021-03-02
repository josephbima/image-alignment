from google_images_download import google_images_download
import os
from utils import imageCropper, splitImagesIntoThree
import cv2
from main import alignImages
from error_calculation import compare_errors
import csv


# creating object
response = google_images_download.googleimagesdownload()

search_queries = [

    '1200x800 images'
]


def downloadimages(query):
    # keywords is the search query
    # format is the image file format
    # limit is the number of images to be downloaded
    # print urs is to print the image file url
    # size is the image size which can
    # be specified manually ("large, medium, icon")
    # aspect ratio denotes the height width ratio
    # of images to download. ("tall, square, wide, panoramic")
    arguments = {"keywords": query,
                 "format": "jpg",
                 "limit": 50,
                 "print_urls": True}
    try:
        response.download(arguments)

        # Handling File NotFound Error
    except FileNotFoundError:
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit": 4,
                     "print_urls": True}

        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments)
        except:
            pass


# Driver Code for image download
# for query in search_queries:
#     downloadimages(query)
#     print()

# Code to simulate downsampling
def simulate_downsampling(directory):
    # Driver code to simulate downsampling
    # directory = r'./downloads/1200x800 images'
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            fullpath = (os.path.join(directory, filename))
            splitImagesIntoThree(fullpath, filename)
            img = cv2.imread(fullpath, cv2.IMREAD_COLOR)

            crop = imageCropper(img,218,777,232,594)
            cv2.imwrite(f'{filename}_truth.jpg', crop)

# #
# directory = r'./downloads/1200x800 images'
# for filename in os.listdir(directory):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         fullpath = (os.path.join(directory, filename))
#
#         other_dir = r'./4-google-datasets'
#         file_1 = f'{filename}_1.jpg'
#         file_2 = f'{filename}_2.jpg'
#         file_3 = f'{filename}_3.jpg'
#         output_12 = f'{filename}_12.jpg'
#         output_123 = f'{filename}_al.jpg'
#
#         config_1 = {
#             'ref': f'{other_dir}/{file_1}',
#             'algn': f'{other_dir}/{file_2}',
#             'matches': 10000,
#             'top': 0.5,
#             'out': output_12,
#             'debug': False
#         }
#
#         # alignImages(config=config_1) # Uncomment to test
#
#         config_2 = {
#             'ref': f'{output_12}',
#             'algn': f'{other_dir}/{file_3}',
#             'matches': 10000,
#             'top': 0.5,
#             'out': output_123,
#             'debug': False
#         }
#
#         # print(file_1)
#         # print(file_2)
#         # print(file_3)
#         # print(output_12)
#         # print(output_123)
#
#         # alignImages(config=config_2) # Uncomment to test
#
#
#     else:
#         continue
# #
# # # Error Calculation
# #
# directory = r'./downloads/1200x800 images'


total_mse = 0
total_ssim = 0
i = 0

csv_arr = [["ground_truth", "generated_img", "mse", "ssim"]]

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):

        imA = cv2.imread(f'./4-google-datasets/{str(filename)}_truth.jpg')
        imB = cv2.imread(f'./results_4/{str(filename)}_al.jpg')

        mse, s = compare_errors(imA, imB, title=filename)

        total_mse += mse
        total_ssim += s

        csv_arr.append([f'{filename}_truth.jpg', f'{filename}_al.jpg', mse, s])
        # break
        i += 1

with open('results_4/error_calculation.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for r in csv_arr:
        writer.writerow(r)
    writer.writerow(['total_average', 'total_average', total_mse / i, total_ssim / i])

print(f'Mean mse: {total_mse / i}')
print(f'Mean ssim: {total_ssim / i}')