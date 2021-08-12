import numpy as np
import cv2




def myID():
    return 312320062


"""
Convolve a 1-D array with a given kernel
:param inSignal: 1-D array
:param kernel1: 1-D array as a kernel
:return: The convolved array
"""


def leftBorder(i, conv, inSignal, kernel1):
    ind = i
    ind2 = len(kernel1) - 1
    sum = 0
    for j in range(i + 1, 0, -1):
        sum += inSignal[ind] * kernel1[ind2]
        ind2 -= 1
        ind -= 1

    conv[i] = sum
    pass


def rightBorder(i, conv, inSignal, kernel1):
    ind = len(inSignal) - (len(conv) - i)
    ind2 = 0
    sum = 0
    for j in range(len(conv) - i):
        sum += inSignal[ind] * kernel1[ind2]
        ind += 1
        ind2 += 1
    conv[i] = sum
    pass


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    conv = np.zeros(len(inSignal) + len(kernel1) - 1)
    fkernel = np.flip(kernel1)
    for i in range(len(conv)):
        if i < len(kernel1) - 1:
            leftBorder(i, conv, inSignal, fkernel)
        elif i > len(inSignal) - 1:
            rightBorder(i, conv, inSignal, fkernel)
        else:
            sum = 0
            ind = i
            ind2 = len(kernel1) - 1
            for j in range(len(kernel1)):
                sum += inSignal[ind] * fkernel[ind2]
                ind -= 1
                ind2 -= 1
            conv[i] = sum
    return conv


"""
Convolve a 2-D array with a given kernel
:param inImage: 2D image
:param kernel2: A kernel
:return the convolved Imaged.
"""


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    rows = kernel2.shape[0]
    cols = kernel2.shape[1]
    midR = np.floor(rows / 2).astype(int)
    midC = np.floor(cols / 2).astype(int)

    padded = np.pad(inImage.astype(np.float32), ((midR, midR), (midC, midC)), 'edge')
    conv = np.zeros_like(inImage)
    for i in range(midR, padded.shape[0] - midR):
        for j in range(midC, padded.shape[1] - midC):
            sx = i - midR
            ex = i + midR + 1
            sy = j - midC
            ey = j + midC + 1

            conv[i - midR, j - midC] = np.sum(padded[sx: ex, sy: ey] * kernel2)

    return conv


"""
Calculate gradient of an image
:param inImage: Grayscale iamge
:return: (directions, magnitude,x_der,y_der)
"""


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    kernel = np.array([[1, 0, -1]])
    x_der = conv2D(inImage, kernel)
    y_der = conv2D(inImage, kernel.transpose())
    mag = np.sqrt(np.power(x_der, 2) + np.power(y_der, 2))
    direction = np.arctan2(y_der, x_der)

    return direction, mag, x_der, y_der


"""
Blur an image using a Gaussian kernel
:param inImage: Input image
2:param kernelSize: Kernel size
:return: The Blurred image
"""


def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    Gkernel = getGKernel(kernel_size)
    bluredImg = conv2D(in_image, Gkernel)
    return bluredImg


# for some reason my Gaussian kernel different from cv Gaussian kernel at 0.001~0.1
def getGKernel(kernel_size: np.ndarray) -> np.ndarray:
    if kernel_size.shape[0] % 2 == 0 or kernel_size.shape[1] % 2 == 0:
        raise ValueError("Gaussian Kernel siz must be odd number")
    kernel = np.array([1, 1]).astype(np.float32)
    Garr = np.array([1, 1]).astype(np.float32)
    for i in range(2, kernel_size.shape[0]):
        Garr = conv1D(Garr, kernel)
    Garr = Garr.reshape(Garr.shape[0], 1)
    Gkernel = Garr @ Garr.transpose()
    Gkernel = Gkernel / np.sum(Gkernel)

    return Gkernel


"""
Blur an image using a Gaussian kernel using OpenCV built-in functions
:param inImage: Input image
:param kernelSize: Kernel size
:return: The Blurred image
"""


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    if kernel_size.shape[0] % 2 == 0 or kernel_size.shape[1] % 2 == 0:
        raise ValueError("Gaussian kernel size must be od number")
    Garr = cv2.getGaussianKernel(kernel_size.shape[0], -1)
    Gkernel = Garr @ Garr.transpose()
    blured = cv2.filter2D(in_image, -1, Gkernel, borderType=cv2.BORDER_REPLICATE)
    return blured


"""
Detects edges using the Sobel method
:param img: Input image
:param thresh: The minimum threshold for the edge response
:return: opencv solution, my implementation
"""


def getSobelDer(img):
    sobel_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Ix = conv2D(img, sobel_filter.transpose())
    Iy = conv2D(img, sobel_filter)
    return Ix, Iy


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    Ix, Iy = getSobelDer(img)
    myMag = np.sqrt(np.power(Ix, 2) + np.power(Iy, 2))
    myMag[myMag >= thresh] = 1
    myMag[myMag <= thresh] = 0

    Cx = cv2.Sobel(img, cv2.CV_64F, 0, 1, thresh)
    Cy = cv2.Sobel(img, cv2.CV_64F, 1, 0, thresh)
    cvMag = cv2.magnitude(Cx, Cy)
    cvMag[cvMag >= thresh] = 1
    cvMag[cvMag < thresh] = 0

    return cvMag, myMag


def zeroCrossing(laplacian):
    new_img = np.zeros_like(laplacian)

    for i in range(laplacian.shape[0]):
        for j in range(laplacian.shape[1] - 1):
            pattern1 = [laplacian[i, j] > 0, laplacian[i, j + 1] < 0]
            if j != 0:
                pattern2 = [laplacian[i, j] == 0, laplacian[i, j - 1] > 0, laplacian[i, j + 1] < 0]
                if all(pattern2):
                    new_img[i, j] = 1

            if all(pattern1):
                new_img[i, j] = 1

    return new_img

    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: :return: Edge matrix
    """


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    smoothed = blurImage2(img, np.ndarray((3, 3)))
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian = conv2D(smoothed, laplacian_filter)
    newImg = zeroCrossing(laplacian)

    return newImg


def non_max_suppression(img, dir):
    suppressed = np.zeros_like(img)
    rows, cols = img.shape
    angles = np.copy(dir)
    angles = angles * 180 / np.pi
    angles[angles < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            try:
                q = 255
                r = 255

                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                elif 22.5 <= angles[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                elif 67.5 <= angles[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                elif 112.5 <= angles[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if img[i, j] >= q and img[i, j] >= r:
                    suppressed[i, j] = img[i, j]
                else:
                    suppressed[i, j] = 0

            except IndexError as e:
                pass

    return suppressed


"""
Detecting edges usint "Canny Edge" method
:param img: Input image
:param thrs_1: T1
:param thrs_2: T2
:return: opencv solution, my implementation
"""


def threshold(supressed, ht, lt):
    M, N = supressed.shape
    res = np.zeros((M, N), dtype=np.int32)
    strong_i, strong_j = np.where(supressed >= ht)
    weak_i, weak_j = np.where((supressed <= ht) & (supressed >= lt))

    res[strong_i, strong_j] = 1
    res[weak_i, weak_j] = 0.5

    return res, 0.5, 1


def hysteresis(thresh, weak, strong):
    rows, cols = thresh.shape
    eps = 3
    for i in range(eps, rows - eps):
        for j in range(eps, cols - eps):
            if thresh[i, j] == weak:
                try:
                    neighbors = thresh[i - eps:i+eps, j-eps:, j+eps]
                    row, col = np.where(neighbors == strong)
                    if len(row) > 0:
                        thresh[i, j] = strong
                    else:
                        thresh[i, j] = 0
                except IndexError as e:
                    pass

    return thresh



def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    img = blurImage2(img, np.ndarray((3, 3)))
    x_der, y_der = getSobelDer(img)
    mag = np.sqrt(np.power(x_der, 2) + np.power(y_der, 2))
    dir = np.arctan2(y_der, x_der)
    img = img * 255
    img = img.astype('uint8')
    supressed = non_max_suppression(mag, dir)
    thresh, weak, strong = threshold(supressed, thrs_1, thrs_2)

    myCanny = hysteresis(thresh, weak, strong)

    cvCanny = cv2.Canny(img, int(thrs_1 * 255), int(thrs_2* 255))
    return cvCanny, myCanny

"""
Find Circles in an image using a Hough Transform algorithm extension
:param I: Input image
:param minRadius: Minimum circle radius
:param maxRadius: Maximum circle radius
:return: A list containing the detected circles,
[(x,y,radius),(x,y,radius),...]
"""


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    fCircles = []
    circle_candidates = np.zeros((img.shape[0], img.shape[1], max_radius + 1), dtype=np.int32)
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0)# get sobel derivetive
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    direction = np.arctan2(Iy, Ix) * 180 / np.pi        # get direction and change to radians
    direction = np.radians(direction)
    img = img * 256
    img = img.astype('uint8')
    canny = cv2.Canny(img, 50, 100)
    # move on all canny img, if the pixel is an edge pixel vote for the possible circle.
    for x in range(0, canny.shape[0]):
        for y in range(0, canny.shape[1]):
            if canny[x, y] > 0:
                for rad in range(min_radius, max_radius + 1):
                    cx1 = int(x + rad * np.cos(direction[x, y] - np.pi / 2))
                    cy1 = int(y - rad * np.sin(direction[x, y] - np.pi / 2))
                    cx2 = int(x - rad * np.cos(direction[x, y] - np.pi / 2))
                    cy2 = int(y + rad * np.sin(direction[x, y] - np.pi / 2))
                    if 0 < cx1 < len(circle_candidates) and 0 < cy1 < len(circle_candidates[0]):
                        circle_candidates[cx1, cy1, rad] += 1
                    if 0 < cx2 < len(circle_candidates) and 0 < cy2 < len(circle_candidates[0]):
                        circle_candidates[cx2, cy2, rad] += 1

    # filter all the circles with less then half from the max circle votes.
    thresh = 0.45 * circle_candidates.max()
    centerB, centerA, radius = np.where(circle_candidates >= thresh)

    # delete similar circles
    eps = 14
    for i in range(0, len(centerA)):  # delete similar circles.
        if centerB[i] == centerA[i] == radius[i] == 0:
            continue
        temp = (centerB[i], centerA[i], radius[i])
        similarCircles = np.where((temp[0] - eps <= centerB) & (centerB <= temp[0] + eps)
                           & (temp[1] - eps <= centerA) & (centerA <= temp[1] + eps)
                           & (temp[2] - eps <= radius) & (radius <= temp[2] + eps))[0]
        for j in range(1, len(similarCircles)):
            centerB[similarCircles[j]] = 0
            centerA[similarCircles[j]] = 0
            radius[similarCircles[j]] = 0

    for i in range(len(centerA)):  # get all the circles that left after delete similar.
        if centerA[i] == centerB[i] == radius[i] == 0:
            continue
        fCircles.append((centerA[i], centerB[i], radius[i]))

    return fCircles

