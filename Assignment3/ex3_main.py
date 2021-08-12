from ex3_utils import *
import time


def lkDemo(img_path):
    print("LK Demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    print(img_1.shape)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    et = time.time()
    print("Time: {:.4f}".format(et - st))
    print(pts)
    print("ub,", uv)
    displayOpticalFlow(img_2, pts, uv)


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrDemo2(img_path):
    img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB) / 255
    lvs = 4
    ga_pyr = gaussianPyr(img, lvs)
    laplaceianReduce(img, lvs)

    gaker = cv2.getGaussianKernel(5,-1)
    gker = gaker @ gaker.transpose()
    gker *= 4
    print("gket sum", np.sum(gker))
    a, f = plt.subplots(1, 2)
    f[0].imshow(ga_pyr[2])
    f[0].set_title('small image')
    f[1].imshow(gaussExpand(ga_pyr[2], gker))
    f[1].set_title('after expand')
    plt.show()

    pass


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def getmask(path, shape):
    print(shape)
    img = cv2.imread(path) / 255
    new_img = np.copy(img)
    new_img[img[:, :, 0] > 0.6] = 255
    new_img[:, 500:] = 0
    print(new_img.shape)
    return new_img




def blendDemo():
    im2 = cv2.cvtColor(cv2.imread('sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im1 = cv2.cvtColor(cv2.imread('cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    if im1.shape > im2.shape:
        shape = im1.shape
    else:
        shape = im2.shape
    mask = getmask('images/cat.jpg', shape) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    img_path = 'images/boxman.jpg'
    lkDemo(img_path)
    pyrGaussianDemo('images/pyr_bit.jpg')
    pyrDemo2(img_path)
    pyrLaplacianDemo('images/cat.jpg')
    blendDemo()


if __name__ == '__main__':
    main()
