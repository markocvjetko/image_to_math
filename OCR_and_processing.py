import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def contour_bounding_box(contour):
    '''
    Returns bounding box from given contour
    :param contours: contour from cv.findContours()
    :return: (min_x, min_y, max_x, max_y)
    '''
    return (min(contour[:, 0]), min(contour[:, 1]), max(contour[:, 0]), max(contour[:, 1]))


if __name__ == '__main__':
    image = cv.imread("/home/marko/photomath_interview/example.jpeg", 0)
    #image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_grey = image[10:-10, 10:-10]
    (thresh, image_bw) = cv.threshold(image_grey, 127, 255, cv.THRESH_BINARY)
    image_bw = 255-image_bw
    contours, _ = cv.findContours(image_bw, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    bounding_boxes = []
    for contour in contours:
        bounding_boxes.append((cv.boundingRect(contour)))
        bounding_boxes.sort()

    for i, bb in enumerate(bounding_boxes):
        print('bb', bb)
        x, y, w, h = bb
        dim = max(w, h)
        mask
        '''
        image_bw = cv.rectangle(image_bw, (x, y), (x + w, y + h), 127, 2)
        crop_img = image_bw[x:x + w, y:y + h]
        print('crop', crop_img.shape)
        dim = max(w, h)
        rect_img = np.zeros((dim, dim))
        if w > h:
            part = rect_img[:, (w-h)//2:(w+h)//2]
            part = part +crop_img
            cv.imwrite((str(i) + '_image.jpeg'), rect_img)
'''
    plt.imshow(rect_img, cmap='gray')
    plt.show()
