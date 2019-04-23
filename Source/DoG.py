import cv2
import numpy as np
import math
import sys
def DOG(image:np.ndarray,sigma = 1.6,threshold = 200):
    image_list = []
    temp_list = []
    binary_list = []
    row = image.shape[0]
    col = image.shape[1]
    k = 1.4
    ksize = 3
    sigma_list = [1.6]
    for time in range(0,8):
        sigma_list.append(k*sigma)
        blur = cv2.GaussianBlur(image,(ksize,ksize),k * sigma)
        image_list.append(blur)
        ksize += 2
        k *= 1.4
    for i in range(0,len(image_list) - 1):
        temp = image_list[i + 1] - image_list[i]
        temp = (temp > 200) * temp
        cv2.imshow('DOG ' + str(i),temp)
        temp_list.append(temp)
    for i in range(0,len(temp_list)):
        # Find x and y derivatives
        dy, dx = np.gradient(temp_list[i])
        Ixx = dx ** 2
        Ixy = dy * dx
        Iyy = dy ** 2
        height = temp_list[i].shape[0]
        width = temp_list[i].shape[1]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Calculate sum of squares
                windowIxx = Ixx[y - 1:y + 2, x - 1:x + 2]
                windowIxy = Ixy[y - 1:y + 2, x - 1:x + 2]
                windowIyy = Iyy[y - 1:y + 2, x - 1:x + 2]
                Sxx = windowIxx.sum()
                Sxy = windowIxy.sum()
                Syy = windowIyy.sum()
                det = (Sxx * Syy) - 0.04 * (Sxy ** 2)
                trace = Sxx + Syy
                if trace == 0 or ((det**2) / trace) > 10:
                    temp_list[i][y, x] = 0
    image_list.clear()
    binary = np.zeros((row, col))
    for i in range(1,6):
        up_image = temp_list[i + 1]
        current_image = temp_list[i]
        down_image = temp_list[i - 1]
        for x in range(1,row - 1):
            for y in range(1,col - 1):
                if ((current_image[x,y] >= current_image[x-1:x+2,y-1:y+2].all() and
                    current_image[x,y] >= up_image[x-1:x+2,y-1:y+2].all() and
                    current_image[x,y] >= down_image[x-1:x+2,y-1:y+2].all()) or
                    (current_image[x, y] <= current_image[x - 1:x + 2, y - 1:y + 2].all() and
                     current_image[x, y] <= up_image[x - 1:x + 2, y - 1:y + 2].all() and
                     current_image[x, y] <= down_image[x - 1:x + 2, y - 1:y + 2].all())):
                    if current_image[x, y] > threshold:
                        binary[x,y] = sigma_list[i]
                    else:
                        current_image[x,y] = 0
    temp_list.clear()
    color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for i in range(1,row - 1):
        for j in range(1,col - 1):
            if binary[i,j] != 0:
                color.itemset((i, j, 0), 0)
                color.itemset((i, j, 1), 0)
                color.itemset((i, j, 2), 255)
    cv2.imshow('output',color)
    cv2.waitKey(0)

def main(argv):
    image = cv2.imread(argv[0], cv2.IMREAD_GRAYSCALE)
    sigma = float(argv[1])
    threshold = int(argv[2])
    DOG(image,sigma,threshold)

if __name__ == '__main__':
    main(sys.argv[1:])