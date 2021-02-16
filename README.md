# Image Alignment

Image alignment and cropping using openCV.

# Updates

### Test 1: Common Area of Two Similar Images

A test of finding the common area for two similar images. The images are of the same size and somewhat same quality. 
The parameters used are 10000 matches and 0.3 top matches.

Visually, the results were good and can move on to more tests. 

### Test 2: Common Area of a HR and a LR Images

Test for common area between one high resolution image and one low resolution one. One image is manually downsampled 
to 0.25 size to simulate real life data. The process are as follows:

1. Upsample low resolution image
2. Get keypoints from low resolution and high resolution image separately
3. Match their descriptors using RANSAC
4. cv2.warpPerspective and crop image

The results were varied, where some images are warped unrecognizable, while some images showed little difference to 
their results from test one. 

### Usage

```
python main.py --ref tis1.jpg --algn tis2.jpg --out align_tis.jpg --matches 20000 --top 0.5 --debug True
python main.py --ref vit2.jpg --algn vit1.jpg --out align_vit.jpg --matches 200 --top 0.15              
```

### Args

```
--ref : reference image path

--algn : shifted image path

--out : saved aligned image name

--matches : number of matches to be observed (default: 200)

--top : top matches (in percentage) that we want to consider (default: 0.15)

--debug : set to True to see matching process

```




