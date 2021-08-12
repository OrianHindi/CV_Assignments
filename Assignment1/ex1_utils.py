"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
yiq_filter = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 312320062

def clacHist(img):
    pass



def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation == LOAD_GRAY_SCALE:
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        normalized = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    elif representation == LOAD_RGB:
        img = cv.imread(filename, cv.IMREAD_COLOR)
        normalized = cv.normalize(cv.cvtColor(img, cv.COLOR_BGR2RGB), None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    return normalized


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if representation == LOAD_GRAY_SCALE:
        img = imReadAndConvert(filename, LOAD_GRAY_SCALE)
        plt.imshow(img, cmap='gray')
        plt.show()

    elif representation == LOAD_RGB:
        img = imReadAndConvert(filename, LOAD_RGB)
        plt.imshow(img)
        plt.show()

    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    if imgRGB.ndim == 3:
        shape = imgRGB.shape
        imgYIQ = np.dot(imgRGB.reshape(-1, 3), yiq_filter.transpose()).reshape(shape)
        return imgYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    shape = imgYIQ.shape
    imgRGB = np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_filter).transpose()).reshape(shape)

    return imgRGB


def histogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if imgOrig.ndim == 2:  # image is in grayscale form
        img = cv.normalize(imgOrig, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    elif imgOrig.ndim == 3:  # image is in rgb form, need to work on Y channel at YIQ space
        img = transformRGB2YIQ(imgOrig)[:, :, 0]
        img = cv.normalize(img, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    histOrg, bins = np.histogram(img, 256)  # calculate image histogram
    cum_sum = np.cumsum(np.array(histOrg))      # calculate cum_sum
    norm_cum_sum = cum_sum / np.max(cum_sum)  # normalize the cum_sum
    lut = (norm_cum_sum * 255).astype('uint8')  # make a map for new intensities.

    imEq = lut[img.reshape(1, -1)]  # map each intensity in the img to new intensity in new img
    imEq = imEq.reshape(imgOrig.shape[0], imgOrig.shape[1])  # reshape new image
    histEq, binsEq = np.histogram(imEq, 256)  # calculate new img histogram.
    if imgOrig.ndim == 2:
        return imEq, histOrg, histEq
    else:  # if we need to return rgb, get the image in yiq space, change Y channel and transform to rgb space.
        yiq_image = transformRGB2YIQ(imgOrig)
        yiq_image[:, :, 0] = cv.normalize(imEq, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        imEq = transformYIQ2RGB(yiq_image)
        imEq.reshape(imgOrig.shape)
        return imEq, histOrg, histEq


def weightedMean(hist, left, right):
    numpix = 0
    sumpix = 0
    for i in range(int(left + 0.5), int(right + 0.5)):
        sumpix += i*hist[i]  # calculate the intensity * num of pixels of this intensity
        numpix += hist[i]  # calculate number of pixel in ea section
    return sumpix/numpix


def update_qvalues(q_values, borders, hist):
    for i in range(0, len(q_values)):
        q = weightedMean(hist, borders[i], borders[i+1])
        q_values[i] = q

    return q_values


def getImage(imgOrig, borders, q_values):
    # Return a new picture with the q values, map every range of pixels into 1 number.
    temp = np.copy(imgOrig)
    for i in range(0, len(borders)-1):
        mask = np.logical_and(imgOrig >= borders[i], imgOrig <= borders[i+1])
        temp[mask] = q_values[i]

    return temp


def changeY(new_img, imOrig):
    # change the Y channel after done calc the mat and return the RGB after changes.
    yiq = transformRGB2YIQ(imOrig)
    yiq[:, :, 0] = new_img
    rgb = transformYIQ2RGB(yiq)

    return rgb


def update_borders(borders, q_values):
    for i in range(1, len(borders)-1):
        borders[i] = np.floor((q_values[i-1] + q_values[i])) / 2

    return borders


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if imOrig.ndim == 3:  # if image in rgb space change to YIQ and work on Y channel
        img = transformRGB2YIQ(imOrig)[:, :, 0]
        img = cv.normalize(img, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)  # normalize img to 0..255 unit8
    elif imOrig.ndim == 2:  # image in grayscale
        img = cv.normalize(imOrig, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)  # normalize img to 0..255 unit8

    images = []
    errors = []
    hist, bins = np.histogram(img, 256)  # calculate img histogram
    borders = np.arange(nQuant+1)  # arrange borders array
    k = np.floor(256/nQuant)  # get borders
    borders = borders * k
    print(borders)
    q_values = np.zeros(nQuant)  # initialize q values array
    borders[len(borders)-1] = 255  # last border is 255
    for i in range(nIter):
        q_values = update_qvalues(q_values, borders, hist)  # update q_values with weightedmean
        new_img = getImage(img, borders, q_values)  # get new image, map each intensity section the 1 value
       # new_img = cv.normalize(new_img, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
        #  / for some reason graph looks worst then the divide by 255
        new_img = new_img / 255

        if imOrig.ndim == 3:  # if rgb convert to YIQ and change Y channle, then insert to images list.
            new_image = changeY(new_img, imOrig)
            images.insert(i, new_image)
        else:  # else its gray scale, normalize to 0..1 and insert into images list
            new_image = new_img
            images.append(new_img)

        error = np.sqrt(np.sum(np.power(imOrig - new_image, 2))) / (
                    img.shape[0] * img.shape[1])  # calculate the error for iteration i
        errors.append(error)
        borders = update_borders(borders, q_values)  # update borders.

    return images, errors

