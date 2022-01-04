import cv2 as cv
import os
from os import path
import math

import image_utils

BORDER_COLOR = 255
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = ROOT_DIR + "/math-examples"

CONTRAST_TRESHOLD = 90


if __name__ == '__main__':

    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.jpg'):
            print('processing ' + filename)
            image = cv.imread(DATA_DIR + '/' + filename)
            image = image_utils.color_format_to_greyscale(image)
            contrast_image = image_utils.contrastify(image, CONTRAST_TRESHOLD)
            contrast_image = image_utils.negative(contrast_image)
            cv.imshow('Original image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()
            bounding_boxes = image_utils.find_bounding_boxes(contrast_image)
            image_bbs = image_utils.draw_bounding_boxes(contrast_image, bounding_boxes)
            cv.imshow('Characters found', image_bbs)
            cv.waitKey(0)
            cv.destroyAllWindows()
            
