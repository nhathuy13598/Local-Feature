import cv2 as cv
import math as m
import numpy as np
import sys


def Gaussian_kernel(ksize: int = 5, sigma: float = 1):
    '''
    Create Gaussian kernel
    :param ksize: kernel size
    :param sigma: Standard deviation
    :return: numpy array, shape ksize x ksize
    '''
    offset = ksize // 2
    kernel = np.zeros((ksize, ksize))
    coeff = 1 / (2 * m.pi * (sigma ** 2))
    normal = int(sigma * 100)
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            coeff2 = -(i ** 2 + j ** 2) / (2 * (sigma ** 2))
            kernel[i + offset, j + offset] = coeff * m.pow(m.e, coeff2)
    sum = kernel.sum()
    kernel = kernel / sum * normal
    kernel = kernel.round()
    return kernel


def Sobel(image, dx=True, dy=False):
    '''
    Tinh dao ham theo Sobel
    Uu tien tinh dx truoc neu dx == False thi ta moi xet den dy
    :param image: Anh goc
    :param dx: Dao ham theo x
    :param dy: Dao ham theo y
    :return: Anh
    '''
    offset = 1
    row = image.shape[0]
    col = image.shape[1]
    result = np.zeros((row, col), np.uint8)
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    if dx == False and dy == False:
        print("Tham so cho ham Sobel bi loi")
        return None
    if dx == True:
        for i in range(offset, row - offset):
            for j in range(offset, col - offset):
                patch = image[i - offset: i + offset + 1, j - offset: j + offset + 1]
                temp = patch * kernel_x
                sum = temp.sum()
                sum = int(abs(sum))
                result[i, j] = sum
        return result
    if dy == True:
        for i in range(offset, row - offset):
            for j in range(offset, col - offset):
                patch = image[i - offset: i + offset + 1, j - offset: j + offset + 1]
                temp = patch * kernel_y
                sum = temp.sum()
                sum = int(abs(sum))
                result[i, j] = sum
        return result


def Gaussian_Blur(image, kernel):
    '''
    Blur image with Gaussian
    :param image:
    :param kernel:
    :return: Image
    '''
    ksize = kernel.shape[0]
    offset = ksize // 2
    row = image.shape[0]
    col = image.shape[1]
    result = image
    ksum = kernel.sum()
    for i in range(offset, row - offset):
        for j in range(offset, col - offset):
            patch = image[i - offset: i + offset + 1, j - offset: j + offset + 1]
            temp = patch * kernel
            sum = temp.sum() / ksum
            result[i, j] = sum
    return result


def Harris(image, blur = False,ksize = 3, sigma = 1.4, k = 0.04, threshold = 1000):
    '''
    Harris detection
    :param image: Anh goc
    :param kernel: Kich thuoc window
    :param k: He so k thuoc (0.04; 0.06)
    :param threshold: Nguong
    :return: Color image with corner point
    '''
    color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if blur == True:
        Gausskernel = Gaussian_kernel(ksize, sigma)
        Gaussian_Blur(image, Gausskernel)
    dx = Sobel(image, True)
    dy = Sobel(image, False, True)
    offset = ksize // 2
    row = image.shape[0]
    col = image.shape[1]
    dxx = dx ** 2
    dyy = dy ** 2
    dxy = dx * dy
    Rmatrix = np.zeros((row, col), dtype=int)
    for i in range(offset, row - offset):
        for j in range(offset, col - offset):
            patch_xx = dxx[i - offset: i + offset + 1, j - offset: j + offset + 1]
            patch_yy = dyy[i - offset: i + offset + 1, j - offset: j + offset + 1]
            patch_xy = dxy[i - offset: i + offset + 1, j - offset: j + offset + 1]
            Ixx = patch_xx.sum()
            Iyy = patch_yy.sum()
            Ixy = patch_xy.sum()
            R = Ixx * Iyy - (Ixy ** 2) - k * (Ixx + Iyy) ** 2
            Rmatrix[i][j] = int(R)
            if R > threshold:
                color[i, j , 0] = 0
                color[i, j, 1] = 0
                color[i, j, 2] = 255
    return color



def main(argv):
    image = cv.imread(argv[0],cv.IMREAD_GRAYSCALE)
    blur = bool(int(argv[1]))
    ksize = int(argv[2])
    sigma = float(argv[3])
    k = float(argv[4])
    threshold = int(argv[5])
    result = Harris(image.copy(), blur, ksize, sigma, k, threshold)
    cv.imshow('Input', image)
    cv.imshow('Output', result)
    cv.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])

image = cv.imread('checkerboard.png', cv.IMREAD_GRAYSCALE)
result = Harris(image,False,threshold = 100)
cv.imshow('input', image)
cv.imshow('output',result)
cv.waitKey(0)
