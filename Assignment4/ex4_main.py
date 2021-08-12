# ps2
import os
import numpy as np
from ex4_utils import *
import cv2
import matplotlib.pyplot as plt


def get_id():
    return 312320062


def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    plt.colorbar()
    plt.show()


def main():
    ## 1-a
    # Read images

    L = cv2.imread('pair0-L.png', 0) / 255.0
    R = cv2.imread('pair0-R.png', 0) / 255.0
    # Display depth SSD
    displayDepthImage(L, R, method=disparitySSD)
    # # Display depth NC
    L1 = cv2.imread('pair1-L.png', 0) / 255
    R1 = cv2.imread('pair1-R.png', 0) / 255
    displayDepthImage(L, R, method=disparityNC)

    src = np.array([[279, 552],
                    [372, 559],
                    [362, 472],
                    [277, 469]])
    dst = np.array([[24, 566],
                    [114, 552],
                    [106, 474],
                    [19, 481]])
    h, error = computeHomography(src, dst)
    print(h, error)
    dst = cv2.imread('billBoard.jpg', 0) / 255.0
    src = cv2.imread('car.jpg', 0) / 255.0

    warpImag(src, dst)


if __name__ == '__main__':
    print(get_id())
    main()
