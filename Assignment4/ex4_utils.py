import numpy as np
import matplotlib.pyplot as plt


"""
img_l: Left image
img_r: Right image
range: Minimun and Maximum disparity range. Ex. (10,80)
k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

return: Disparity map, disp_map.shape = Left.shape
"""


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    if disp_range[1] - disp_range[0] + 1 > 80:
        raise ValueError("display_range must be lower then 80")
    disp_map = np.zeros(img_l.shape)
    for i in range(k_size, img_l.shape[0]-k_size):
        for j in range(k_size, img_l.shape[1]-k_size):
            sxl = i - k_size
            exl = i + k_size + 1
            syl = j - k_size
            eyl = j + k_size + 1
            patch_left = img_l[sxl:exl, syl:eyl]
            min_ssd = np.inf
            for offset in range(disp_range[0], disp_range[1]):
                sxrr = i - k_size
                exrr = i + k_size + 1
                syrr = j - k_size + offset
                eyrr = j + k_size + 1 + offset

                sxrl = i - k_size
                exrl = i + k_size + 1
                syrl = j - k_size - offset
                eyrl = j + k_size + 1 - offset
                patch_right_r = img_r[sxrr:exrr, syrr:eyrr]
                patch_right_l = img_r[sxrl:exrl, syrl:eyrl]
                if patch_right_r.shape == patch_left.shape:
                    ssd = np.sum((patch_left - patch_right_r) ** 2)
                    if ssd < min_ssd:
                        disp_map[i, j] = offset
                        min_ssd = ssd
                if patch_right_l.shape[0] == patch_left.shape[0] and patch_right_l.shape[1] == patch_left.shape[1]:
                    ssd = np.sum((patch_left - patch_right_l) ** 2)
                    if ssd < min_ssd:
                        disp_map[i, j] = offset
                        min_ssd = ssd
    return disp_map


"""

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """


def normalize_correlation(patch_left, patch_right):
    left_avg = np.average(patch_left)
    right_avg = np.average(patch_right)
    n = patch_left.shape[0] * patch_left.shape[1]
    left_std = np.std(patch_left)
    right_std = np.std(patch_right)
    return 1/n * np.sum(((patch_left - left_avg) * (patch_right - right_avg)) / (left_std * right_std))


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    disp_map = np.zeros(img_l.shape)
    for i in range(k_size, img_l.shape[0] - k_size):
        for j in range(k_size, img_l.shape[1] - k_size):
            sxl = i - k_size
            exl = i + k_size + 1
            syl = j - k_size
            eyl = j + k_size + 1
            patch_left = img_l[sxl:exl, syl:eyl]
            max_ncc = -np.inf
            for offset in range(disp_range[0], disp_range[1]):
                sxrr = i - k_size
                exrr = i + k_size + 1
                syrr = j - k_size + offset
                eyrr = j + k_size + 1 + offset

                sxrl = i - k_size
                exrl = i + k_size + 1
                syrl = j - k_size - offset
                eyrl = j + k_size + 1 - offset
                patch_right_r = img_r[sxrr:exrr, syrr:eyrr]
                patch_right_l = img_r[sxrl:exrl, syrl:eyrl]
                if patch_right_r.shape == patch_left.shape:
                    ncc = normalize_correlation(patch_left, patch_right_r)
                    if ncc > max_ncc:
                        disp_map[i, j] = offset
                        max_ncc = ncc
                if patch_right_l.shape == patch_left.shape:
                    ncc = normalize_correlation(patch_left, patch_right_l)
                    if ncc > max_ncc:
                        disp_map[i, j] = offset
                        max_ncc = ncc
    return disp_map
"""
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
"""


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    if src_pnt.shape[0] < 4 or dst_pnt.shape[0] < 4:
        raise ValueError("Only 4 points and above allowed")
    A = np.zeros((8, 9))
    idx = 0
    for i in range(0, src_pnt.shape[0]):
        x1 = src_pnt[i, 0]
        y1 = src_pnt[i, 1]

        x2 = dst_pnt[i, 0]
        y2 = dst_pnt[i, 1]

        A[idx] = np.array([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1*x2, -x2])
        A[idx + 1] = np.array([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

        idx += 2

    u, s, vh = np.linalg.svd(A)
    L = vh[-1, :] / vh[-1, -1]
    L = L.reshape((3, 3))
    pred = np.zeros((4, 2))
    for j in range(0, src_pnt.shape[0]):
        temp = np.array([src_pnt[j, 0], src_pnt[j, 1], 1]).reshape((3, 1))
        temp2 = np.dot(L, temp)
        pred[j, 0] = temp2[0, 0] / temp2[2, 0]
        pred[j, 1] = temp2[1, 0] / temp2[2, 0]
    return L, np.sqrt(np.sum((pred-dst_pnt))**2)


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output:
        None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()
    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()

    dst_p.append([272, 795])
    dst_p.append([1667, 705])
    dst_p.append([1669, 148])
    dst_p.append([272, 299])
    dst_p = np.array(dst_p)

    ##### Your Code Here ######

    # I had a problem with the plt.imshow, i didnt able to click on the picture. took the points from classmate.

    src_pnt = np.array([[src_img.shape[0], 0], [src_img.shape[0], src_img.shape[1]], [0, src_img.shape[1]], [0, 0]])
    H, mae = computeHomography(src_pnt, dst_p)
    proj_img = np.zeros_like(dst_img)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            temp = np.array([i, j, 1])
            temp2 = H.dot(temp.transpose())
            temp2 /= temp2[2]
            proj_img[int(temp2[1]), int(temp2[0])] = src_img[i, j]

    mask = proj_img == 0
    canvas = dst_img * mask + (1 - mask) * proj_img
    plt.figure()
    plt.imshow(canvas, 'gray')
    plt.figure()
    plt.imshow(proj_img, 'gray')
    plt.show()
    pass
