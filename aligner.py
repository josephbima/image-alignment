from __future__ import print_function
import numpy as np
import cv2
from utils import sampleImage, removeBlackBorders, getRatio, get_resolution, imageCropper, splitImagesIntoThree
from error_calculation import compare_errors
import os
import csv

class Aligner():

    def __init__(self, dir=None, config=None):
        if config is None:
            config = {
                'top': 0.5,
                'matches': 10000
            }

        if dir is None:
            dir = ''

        self.dir = dir
        self.config = config
        self.csv_arr = [["ground_truth", "generated_img", "mse", "ssim"]]


    def set_config(self, config):
        self.config = config

    def get_csv(self):
        return self.csv_arr

    def save_csv(self, total_mse, total_ssim, i, out = 'error_calculation.csv'):

        with open(f'{self.dir}/{out}', 'w', newline='') as file:
            writer = csv.writer(file)
            for r in self.csv_arr:
                writer.writerow(r)
            writer.writerow(['total_average', 'total_average',total_mse/i,total_ssim/i ])

    #  Solve 2 Images
    """
    im1: cv2.img
    im2: cv2.img
    out: str
        output string to save image
    """
    def __solve_img(self, im1, im2, out):

        # We want to use the biggest height and the biggest width for our output image
        height = max(im1.shape[0], im2.shape[0])
        width = max(im1.shape[1], im2.shape[1])

        # We want to resize the smaller image to the bigger image to highlight more features
        if im1.shape[1] * im1.shape[0] > im2.shape[1] * im2.shape[0]:
            im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            im1 = cv2.resize(im1, (im2.shape[1], im2.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Use SIFT to detect features
        sift = cv2.xfeatures2d_SIFT.create()

        keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.config['top'])
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        im1Reg = cv2.warpPerspective(im1, h, (height, width), flags=cv2.INTER_LINEAR)

        # cv2.imshow('with borders', im1Reg)

        cropped_img = removeBlackBorders(im1Reg)

        # cv2.imwrite(os.path.join(f'{self.dir}/result_images/',f'{out}'), cropped_img)

        cv2.destroyAllWindows()

        return cropped_img

    def solve_images(self, img_arr, out):
        output_img = self.__solve_img(img_arr[0], img_arr[1], f'{out}_1.jpg')
        for i in range(1, len(img_arr)):
            output_img = self.__solve_img(output_img, img_arr[i], f'{out}_{i}.jpg')
        return output_img

    def filename_to_img_arr(self, fn_arr):
        print(f'{self.dir}/testing_images/{fn_arr[0]}.jpg')

        return [cv2.imread(f'{self.dir}/testing_images/{x}.jpg', cv2.IMREAD_COLOR) for x in fn_arr]

    # Code to test algorihtm on all pictures in directory
    def test_algorithm(self, simulate=False):
        file_set = set()
        for filename in os.listdir(f'{self.dir}/original_images'):
            if filename.endswith(".jpg"):
                key_name = filename[:-4]
                file_set.add(key_name)
                print(f'Inspecting {filename}')
                if simulate:
                    print('Simulating downsampling...')
                    fullpath = (os.path.join(f'{self.dir}/original_images', filename))
                    print(fullpath)

                    splitImagesIntoThree(fullpath, key_name, f'{self.dir}/testing_images')
                    img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
                    crop = imageCropper(img, 218, 777, 232, 594)

                    cv2.imwrite(os.path.join(f'{self.dir}/truth_images',f'{key_name}_truth.jpg'), crop)

                # End simulation

        total_mse = 0
        total_ssim = 0

        # Run the algorithm on all the files and save them
        for name in file_set:
            print(name)
            file_solve = [f'{name}_{i}' for i in range(1,4)]
            img_solve = self.filename_to_img_arr(file_solve)
            output = aligner.solve_images(img_solve, name)

            # Save the image to the correct directory
            cv2.imwrite(os.path.join(f'{self.dir}/result_images/', f'{name}_al.jpg'), output)

            # cv2.imshow('output', output)
            # cv2.waitKey(0)

            # Compute the error
            ground_truth = cv2.imread(f'{self.dir}/truth_images/{name}_truth.jpg')

            # cv2.imshow('t',ground_truth)
            # cv2.waitKey(0)

            mse, s = compare_errors(ground_truth,output, dir=self.dir, title=name)

            total_mse += mse
            total_ssim += s

            self.csv_arr.append([f'{self.dir}/truth_images/{name}_truth.jpg', f'{self.dir}/result_images/{name}_al.jpg', mse, s])

        self.save_csv(total_mse=total_mse,total_ssim=total_ssim,i=len(file_set))

        cv2.destroyAllWindows()



aligner = Aligner(dir='./testing-dataset')

files = ['city_1.jpg', 'city_3.jpg', 'city_2.jpg']

# aligner.solve_images(aligner.filename_to_img_arr(files))
aligner.test_algorithm()

cv2.destroyAllWindows()