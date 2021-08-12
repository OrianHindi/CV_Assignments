import sys
from typing import List
import numpy as np
import cv2
import numpy.linalg
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

"""
   Given two images, returns the Translation from im1 to im2
   :param im1: Image 1
   :param im2: Image 2
   :param step_size: The image sample size:
   :param win_size: The optical flow window size (odd number)
   :return: Original points [[x,y]...], [[dU,dV]...] for each points
   """

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    origin_index = []
    u_v = []
    if im1.ndim > 2 or im2.ndim > 2:
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    else:
        im1Gray = im1
        im2Gray = im2

    filter = np.array([[1, 0, -1]])
    Ix = cv2.filter2D(im1Gray, -1, filter, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im1Gray, -1, filter.transpose(), borderType=cv2.BORDER_REPLICATE)
    It = im2Gray - im1Gray
    mid = np.round(win_size / 2).astype(int)
    for x in range(mid, im1Gray.shape[0] - mid, step_size):
        for y in range(mid, im1Gray.shape[1] - mid, step_size):
            sx = x - mid
            ex = x + mid + 1
            sy = y - mid
            ey = y + mid + 1

            A = np.zeros((win_size * win_size, 2))
            b = np.zeros((win_size * win_size, 1))

            A[:, 0] = Ix[sx: ex, sy: ey].flatten()
            A[:, 1] = Iy[sx: ex, sy: ey].flatten()
            b[:, 0] = -It[sx: ex, sy: ey].flatten()

            b = A.transpose() @ b
            eign_vals, v = np.linalg.eig(A.transpose() @ A)
            A = A.transpose() @ A

            eign_vals.sort()
            big = eign_vals[1]
            small = eign_vals[0]

            if big >= small > 1 and big / small < 100:
                ans = np.dot(numpy.linalg.pinv(A), b)
                origin_index.append([y, x])
                u_v.append(ans)

    return np.array(origin_index).reshape(len(origin_index), 2), -np.array(u_v).reshape(len(u_v), 2)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaker = cv2.getGaussianKernel(5, -1)
    upsmaple_kernel = gaker @ gaker.transpose()
    upsmaple_kernel *= 4
    gau_pyr = gaussianPyr(img, levels)
    gau_pyr.reverse()
    lap_pyr = [gau_pyr[0]]
    for i in range(1, len(gau_pyr)):
        expanded = gaussExpand(gau_pyr[i-1], upsmaple_kernel)
        if gau_pyr[i].shape != expanded.shape:
            x = expanded.shape[0] - gau_pyr[i].shape[0]
            y = expanded.shape[1] - gau_pyr[i].shape[1]
            expanded = expanded[x::, y::]
            diff_img = gau_pyr[i] - expanded
        else:
            diff_img = gau_pyr[i] - expanded
        lap_pyr.append(diff_img)

    lap_pyr.reverse()
    return lap_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    gaker = cv2.getGaussianKernel(5, -1)
    upsmaple_kernel = gaker @ gaker.transpose()
    upsmaple_kernel *= 4
    lap_pyr.reverse()
    r_img = [lap_pyr[0]]
    for i in range(1, len(lap_pyr)):
        temp = gaussExpand(r_img[i - 1], upsmaple_kernel)
        if lap_pyr[i].shape != temp.shape:
            x = temp.shape[0] - lap_pyr[i].shape[0]
            y = temp.shape[1] - lap_pyr[i].shape[1]
            new_img = temp[x::, y::] + lap_pyr[i]
        else:
            new_img = temp + lap_pyr[i]
        r_img.append(new_img)
    lap_pyr.reverse()
    r_img.reverse()
    return r_img[0]


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    gauss_pyramid = [img]
    gArr = cv2.getGaussianKernel(5, -1)
    gKernel = gArr @ gArr.transpose()
    for i in range(1, levels):
        It = cv2.filter2D(gauss_pyramid[i-1], -1, gKernel)
        It = It[::2, ::2]
        gauss_pyramid.append(It)

    return gauss_pyramid


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    if img.ndim == 3:
        expanded = np.zeros((2 * img.shape[0], 2 * img.shape[1], img.shape[2]))
    else:
        expanded = np.zeros((2 * img.shape[0], 2 * img.shape[1]))
    expanded[::2, ::2] = img
    expanded = cv2.filter2D(expanded, -1, gs_k, borderType=cv2.BORDER_REPLICATE)
    return expanded


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    naive_blend = img_1 * mask + img_2 * (1 - mask)
    img_1_lp = laplaceianReduce(img_1, levels)
    img_2_lp = laplaceianReduce(img_2, levels)
    mask_lp = gaussianPyr(mask, levels)
    img_2_lp.reverse()
    img_1_lp.reverse()
    mask_lp.reverse()
    r_imgs = []
    for i in range(0, len(img_2_lp)):
        new_img = mask_lp[i] * img_1_lp[i] + (1-mask_lp[i]) * img_2_lp[i]
        r_imgs.append(new_img)
    r_imgs.reverse()

    return naive_blend, laplaceianExpand(r_imgs)


