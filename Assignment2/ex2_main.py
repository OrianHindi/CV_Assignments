from ex2_utils import *
import matplotlib.pyplot as plt
import time


def conv1Demo():
    print("Conv1Demo:")
    kernel = np.array([3, 4, 5])
    vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    my_conv = conv1D(vec, kernel)
    np_conv = np.convolve(vec, kernel, 'full')
    print("My conv:", my_conv)
    print("np conv:", np_conv)
    print("-----------------------------------------------------")
    pass


def conv2Demo():
    print("Conv2Demo:")
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('uint8')
    kernel = np.array([[1, 0, -1], [2, 1, -2], [3, 2, -3]])
    my_conv = conv2D(mat, kernel)
    cv_conv = cv2.filter2D(mat, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    print("My conv:", my_conv)
    print("cv conv:", cv_conv)
    print("-----------------------------------------------------")
    pass


def derivDemo():
    plt.gray()
    img = cv2.imread('zebra.jpeg', cv2.IMREAD_GRAYSCALE) / 255
    directions, mag, x_der, y_der = convDerivative(img)
    a, f = plt.subplots(2, 2)
    f[0, 0].set_title('Directions')
    f[0, 0].imshow(directions)
    f[0, 1].imshow(mag)
    f[0, 1].set_title('Magnitude')
    f[1, 0].imshow(x_der)
    f[1, 0].set_title('x_der')
    f[1, 1].set_title('y_der')
    f[1, 1].imshow(y_der)
    plt.show()
    pass


def blurDemo():
    img = cv2.imread('beach.jpg', cv2.IMREAD_GRAYSCALE) / 255
    plt.gray()
    plt.imshow(img)
    plt.title("Real img")
    plt.figure()
    kernel1 = np.ndarray((35, 35))
    my_x5_kernel_blur = blurImage1(img, kernel1)
    cv_x5_kernel_blur = blurImage2(img, kernel1)
    plt.figure()
    plt.imshow(my_x5_kernel_blur)
    plt.title("My blur 35x35 kernel")
    plt.figure()
    plt.imshow(cv_x5_kernel_blur)
    plt.title("cv blur 35x35 kernel")
    plt.show()
    pass


def edgeDemo():
    plt.gray()
    img = cv2.imread('edgespic.jpeg', cv2.IMREAD_GRAYSCALE) / 255
    cv_sobel, my_sobel = edgeDetectionSobel(img, 0.7)
    LOG_edges = edgeDetectionZeroCrossingLOG(img)
    cv_canny, my_canny = edgeDetectionCanny(img, 0.4, 0.7)
    a, f = plt.subplots(1, 2)
    f[0].imshow(cv_sobel)
    f[0].set_title('CV Sobel solution')
    f[1].imshow(my_sobel)
    f[1].set_title('My Sobel solution')
    plt.figure()
    plt.imshow(LOG_edges)
    plt.title('ZerocrossingLOG')
    b, g = plt.subplots(1, 2)
    g[0].imshow(cv_canny)
    g[0].set_title('CV solution Canny')
    g[1].imshow(my_canny)
    g[1].set_title('My solution Canny')
    plt.show()
    pass


def houghDemo():
    img = cv2.imread('circles.jpg', cv2.IMREAD_GRAYSCALE) / 255
    plt.gray()
    plt.imshow(img)
    plt.title('real image')
    circles = houghCircle(img, 10, 150)
    for x, y, r in circles:
        cv2.circle(img, (x, y), r, (0, 0, 255), 2)

    plt.figure()
    plt.imshow(img)
    plt.title('Hough circles')
    plt.show()
    print("Circles list")
    print(circles)
    pass


def main():
    print(myID())
    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()


if __name__ == '__main__':
    main()
