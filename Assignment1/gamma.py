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
from ex1_utils import *


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    if rep == LOAD_RGB:
        img = cv.imread(img_path)
    elif rep == LOAD_GRAY_SCALE:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    cv.namedWindow('Gamma correction')
    temp = img
  #  cv.imshow('Gamma correction', img)

    def changeGamma(x: int):
        correction = float(x)
        correction = correction / 100
        img = 255 * np.power(temp / 255, correction)
        img = img.astype('uint8')
        cv.imshow('Gamma correction', img)

        print("Gamma: ", correction)
        pass

    cv.createTrackbar('gamma', 'Gamma correction', 0, 200, changeGamma)

    changeGamma(0)
    cv.waitKey(0)

    cv.destroyAllWindows()



def main():
    gammaDisplay('images/testImg1.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
