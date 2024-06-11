import cv2
from imutils import contours
import numpy as np
import imutils
import random
import pickle
from skimage.morphology import reconstruction

chars_to_detect = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                   'n', 'o', 'u', 'p', 'r', 's', 't', 'u', 'w', 'v', 'y', 'q', 'z',
                   'x', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# def reconstruction(marker, mask, structural_element):
#     prev_marker = np.zeros_like(marker)
#     while True:
#         marker_dilate = cv2.dilate(marker, structural_element, dst=marker)  # In-place dilation
#         np.bitwise_and(mask, marker_dilate, out=marker)  # In-place bitwise AND
#         if np.array_equal(prev_marker, marker):
#             break
#         np.copyto(prev_marker, marker)  # In-place copy
#     return marker


# def reconstruction(marker, mask, structural_element):
#   prev_marker = np.zeros_like(marker)
#   while not np.array_equal(prev_marker, marker):
#     prev_marker = np.copy(marker)
#     marker_dilate = cv2.dilate(prev_marker, structural_element)
#     marker = np.bitwise_and(mask, marker_dilate)
#
#   return marker

def detect_characters(img_path):
    img = cv2.imread(img_path)

    w, h, _ = img.shape
    padding = 0.15
    img = img[int(padding * w):int(w - padding * w), int(padding * h):int(h - padding * h)]

    # plt.imshow(img)
    # plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # plt.imshow(thresh1, cmap='gray')
    # plt.title('otsu')
    # plt.show()

    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # erode_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    #
    # eroded_image = cv2.erode(thresh1, erode_kernel, iterations=1)
    # plt.imshow(eroded_image, cmap='gray')
    # plt.title('Eroded')
    # plt.show()

    # dilation = cv2.dilate(eroded_image, dilate_kernel, iterations = 1)

    ####

    image_erode = cv2.erode(thresh1, np.ones((51, 1)))

    #dilation = reconstruction(image_erode, thresh1, np.ones((3, 3)))

    dilation = reconstruction(image_erode, thresh1, method='dilation')

    ####
    contours, hierarchy = cv2.findContours(np.uint8(dilation), cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    #
    # plt.imshow(dilation, cmap='gray')
    # plt.title('Dilation')
    # plt.show()

    im2 = img.copy()

    def get_contour_x(c):
        x, y, w, h = cv2.boundingRect(c)
        return x

    contours_sorted = sorted(contours, key=get_contour_x)

    characters = {}
    for cnt in contours_sorted:
        x, y, w, h = cv2.boundingRect(cnt)
        if 500 > w > 100 and 600 > h > 200 and 1 > w / h > 0.2:
            cropped = im2[y:y + h, x:x + w]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(cropped, 120, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary_image = cv2.dilate(binary_image, kernel, iterations=1)
            resized_image = cv2.resize(binary_image, (120, 120))
            avg_value = np.mean(resized_image)
            if avg_value > 120:
                characters[(x, w)] = resized_image

    characters_to_remove = []
    for x, w in characters.keys():
        for x2, w2 in characters.keys():
            if x2 != x and w2 != w:
                if x2 >= x and x2 <= x+w and x2+w2 <= x + w:
                    characters_to_remove.append((x2, w2))

    characters_to_remove = list(set(characters_to_remove))
    for key in characters_to_remove:
        del characters[key]

    characters = {k: v for k, v in sorted(characters.items(), key=lambda item: item[0][0])}
    characters = np.array(list(characters.values()))

    return characters
