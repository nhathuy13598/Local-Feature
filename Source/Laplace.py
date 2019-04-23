import cv2
import numpy as np
import sys
from scipy import ndimage
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
    result = np.zeros((row, col), np.int)
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
                result[i, j] = sum
        return result
    if dy == True:
        for i in range(offset, row - offset):
            for j in range(offset, col - offset):
                patch = image[i - offset: i + offset + 1, j - offset: j + offset + 1]
                temp = patch * kernel_y
                sum = temp.sum()
                result[i, j] = sum
        return result

def Laplace(image:np.ndarray,sigma = 1.6,threshold = 200):
    sigma_list =[sigma]
    k = 1.4
    ksize = 3
    laplace_image = []
    row = image.shape[0]
    col = image.shape[1]
    color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for i in range(0,9):
        temp = image.copy()
        blur_image = cv2.GaussianBlur(image,(ksize,ksize),sigma)
        dx = Sobel(blur_image)
        dxx = Sobel(dx)
        dy = Sobel(blur_image,False,True)
        dyy = Sobel(dy,False,True)
        laplace = dxx + dyy
        for x in range(1,row - 1):
            for y in range(1,col - 1):
                if (laplace[x, y + 1] * laplace[x, y - 1] < 0 or
                    laplace[x - 1, y] * laplace[x + 1, y] < 0 or
                    laplace[x-1, y-1] * laplace[x + 1, y+1] < 0 or
                    laplace[x+1, y-1] * laplace[x-1, y+1] < 0):
                    laplace[x,y] = 0
        laplace = (laplace >= 0) * laplace
        laplace = laplace * (sigma ** 2)
        laplace = (laplace <= 255) * laplace
        laplace = np.uint8(laplace)
        sigma = k * sigma
        sigma_list.append(sigma)
        laplace_image.append(laplace)
        ksize += 2
        cv2.imshow('Laplace scale ' + str(i),laplace)
    row = image.shape[0]
    col = image.shape[1]
    binary = np.zeros((row,col))
    for i in range(1,8):
        up_image = laplace_image[i + 1]
        current_image = laplace_image[i]
        down_image = laplace_image[i - 1]
        for x in range(1,row - 1):
            for y in range(1,col - 1):
                if ((current_image[x,y] >= current_image[x-1:x+2,y-1:y+2].all() and
                    current_image[x,y] >= up_image[x-1:x+2,y-1:y+2].all() and
                    current_image[x,y] >= down_image[x-1:x+2,y-1:y+2].all()) or
                    (current_image[x, y] <= current_image[x - 1:x + 2, y - 1:y + 2].all() and
                     current_image[x, y] <= up_image[x - 1:x + 2, y - 1:y + 2].all() and
                     current_image[x, y] <= down_image[x - 1:x + 2, y - 1:y + 2].all())):
                    if current_image[x, y] > threshold:
                        binary[x, y] = current_image[x, y]
                    else:
                        current_image[x,y] = 0
    for i in range(1,row - 1):
        for j in range(1,col - 1):
            if binary[i,j] != 0:
                color.itemset((i, j, 0), 0)
                color.itemset((i, j, 1), 0)
                color.itemset((i, j, 2), 255)
    return color

def main(argv):
    image = cv2.imread(argv[0],cv2.IMREAD_GRAYSCALE)
    sigma = float(argv[1])
    threshold = int(argv[2])
    color = Laplace(image,sigma,threshold)
    cv2.imshow('Result',color)
    cv2.waitKey(0)
if __name__ == '__main__':
    main(sys.argv[1:])