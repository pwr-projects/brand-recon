from __future__ import print_function

import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('data/logos/orlen.png',0)          # queryImage
img2 = cv2.imread('data/photos/orlen_img_04.jpg',0) # trainImage
img1 = cv2.equalizeHist(img1)
img2 = cv2.equalizeHist(img2)
#img1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#img2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

# Initiate SIFT detector
surf = cv2.xfeatures2d_SURF.create(hessianThreshold=100,
                                    nOctaves=4,
                                    nOctaveLayers=4,
                                    extended=True)

# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.4*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3)
plt.show()

# This tutorial code's is shown lines below. You can also download it from here


'''parser = argparse.ArgumentParser(description='Code for Feature Detection tutorial.')
parser.add_argument('--input', help='Path to input image.', default='data/logos/orlen.png')
args = parser.parse_args()
src = cv2.imread('data/photos/lot_img_30.jpg',0)  # cv2.imread(cv2.samples.findFile(args.input), cv2.IMREAD_GRAYSCALE)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
#-- Step 1: Detect the keypoints using SURF Detector
minHessian = 400
detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints = detector.detect(src)
#-- Draw keypoints
img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
cv2.drawKeypoints(src, keypoints, img_keypoints)
#-- Show detected (drawn) keypoints
cv2.imshow('SURF Keypoints', img_keypoints)
cv2.waitKey()'''