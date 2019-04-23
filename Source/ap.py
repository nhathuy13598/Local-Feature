import cv2 as cv
import math as m
import numpy as np
from scipy import signal
def Gaussian_kernel(ksize:int = 5,sigma:float = 1):
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

def Sobel(image,dx = True, dy = False):
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
    kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
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

def Gaussian_Blur(image,kernel):
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
    for i in range(offset,row - offset):
        for j in range(offset, col - offset):
            patch = image[i - offset: i + offset + 1, j - offset: j + offset + 1]
            temp = patch * kernel
            sum = temp.sum() / ksum
            result[i, j] = sum
    return result

kernel = Gaussian_kernel(5,1.4)
dx = Sobel(kernel,True,False)
dxx = Sobel(kernel,True,False)
dy = Sobel(kernel,False,True)
dyy = Sobel(kernel,False,True)
d = dxx + dyy
print(d)