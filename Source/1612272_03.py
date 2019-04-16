import cv2 as cv
from sklearn.feature_extraction import image
from PIL import Image as Img
import numpy as np

a = cv.imread("checkerboard.png", cv.IMREAD_GRAYSCALE)
row = a.shape[0]
col = a.shape[1]
#grad_y,grad_x = np.gradient(a)
#cv.imshow("input",a)
# Đạo hàm theo x
grad_x = cv.Sobel(src=a, ddepth=cv.CV_8UC1, dx=1, dy=0, ksize=3)
# Đạo hàm theo y
grad_y = cv.Sobel(src=a, ddepth=cv.CV_8UC1, dx=0, dy=1, ksize=3)
# Đạo hàm theo xx
grad_xx = grad_x ** 2
# Đạo hàm theo yy
grad_yy = grad_y ** 2
# Đạo hàm theo xy
grad_xy = grad_x * grad_y

# Image patch
ksize = 5
patch_xx = image.extract_patches_2d(grad_xx,(ksize,ksize))
patch_yy = image.extract_patches_2d(grad_yy,(ksize,ksize))
patch_xy = image.extract_patches_2d(grad_xy,(ksize,ksize))
patch_size = patch_xx.shape[0]
R = np.empty((0,0),dtype=int)
# for i in range(patch_size):
#     M = np.zeros((2,2))
#     #I_xx = patch_x[i]*patch_x[i]
#     #I_xx = cv.GaussianBlur(I_xx,(ksize,ksize),1)
#     #M[0,0] = np.mean(I_xx)
#     M[0,0] = patch_xx[i].sum()
#     #I_yy = patch_y[i]*patch_y[i]
#     #I_yy = cv.GaussianBlur(I_yy,(ksize,ksize),1)
#     #M[1,1] = np.mean(I_yy)
#     M[1,1] = patch_yy[i].sum()
#     #I_xy = patch_x[i]*patch_y[i]
#     #I_xy = cv.GaussianBlur(I_xy,(ksize,ksize),1)
#     #M[0,1] = M[1,0] = np.mean(I_xy)
#     M[0,1] = M[1,0] = patch_xy[i].sum()
#     r = (M[0,0]*M[1,1] - M[0,1]**2) - 0.04*((M[0,0]+M[0,1])**2)
#     r = round(r)
#     if r > 0:
#         R = np.append(R,255)
#         #print("Yes")
#     else:
#         R = np.append(R,0)
# offset = 2
# for y in range(offset, row-offset):
#         for x in range(offset, col-offset):
#             #Calculate sum of squares
#             windowIxx = grad_xx[y-offset:y+offset+1, x-offset:x+offset+1]
#             windowIxy = grad_xy[y-offset:y+offset+1, x-offset:x+offset+1]
#             windowIyy = grad_yy[y-offset:y+offset+1, x-offset:x+offset+1]
#             Sxx = windowIxx.sum()
#             Sxy = windowIxy.sum()
#             Syy = windowIyy.sum()
#
#             #Find determinant and trace, use to get corner response
#             det = (Sxx * Syy) - (Sxy**2)
#             trace = Sxx + Syy
#             r = det - 0.04*(trace**2)
#             if r > 0:
#                 R = np.append(R,255)
#                 #print("Yes")
#             else:
#                 R = np.append(R,0)

# In hình ảnh
# cv.imshow("Check board", a)
# cv.imshow("Gradient x", grad_x)
# cv.imshow("Gradient y", grad_y)
# cv.imshow("Gradient xx", grad_xx)
# cv.imshow("Gradient yy", grad_yy)
# cv.imshow("Gradient xy", grad_xy)
def findCorners(img, window_size, k, thresh):
    """
    Finds and returns list of corners and new image with corners drawn
    :param img: The original image
    :param window_size: The size (side length) of the sliding window
    :param k: Harris corner constant. Usually 0.04 - 0.06
    :param thresh: The threshold above which a corner is counted
    :return:
    """
    #Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]

    cornerList = np.empty((0,0))
    newImg = img.copy()
    color_img = cv.cvtColor(newImg, cv.COLOR_GRAY2RGB)
    offset = (int)(window_size/2)

    #Loop through image and find our corners

    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            #Calculate sum of squares
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            #If corner response is over threshold, color the point and add to corner list
            if r > thresh:
                cornerList = np.append(cornerList, r)
                color_img.itemset((y, x, 0), 0)
                color_img.itemset((y, x, 1), 0)
                color_img.itemset((y, x, 2), 255)
            else:
                cornerList = np.append(cornerList, 0)
    return color_img, cornerList

Result,R = findCorners(a,5,0.04,1000)
R = R.reshape((col - 4,-1))
print("Shape cua R:",R.shape)
R = Img.fromarray(R,'P')
R.show()
cv.imshow("R",Result)
#cv.imshow("R",R)
cv.waitKey(0)
