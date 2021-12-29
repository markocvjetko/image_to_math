import pickle
import cv2 as cv
import os
import math
import torch
from NeuralNetwork import NeuralNetwork
from math_parser import parse_expression

BORDER_COLOR = 255
CONTRAST_TRESHOLD = 130
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = ROOT_DIR + '/image-examples/3.jpg'
MODEL_PATH = ROOT_DIR + '/model.pth'
CLASS_DICT_PATH = ROOT_DIR + '/class-ids.txt'


def color_format_to_greyscale(image):
    if len(image.shape) == 3:  # shape 3 for colored images, 2 for greyscale
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        return image


def contrastify(image):
    (thresh, image_bw) = cv.threshold(image, CONTRAST_TRESHOLD, 255, cv.THRESH_BINARY)
    return 255 - image_bw  # invert colors


def find_bounding_boxes(image, cutoff=40):
    contours, _ = cv.findContours(image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    bounding_boxes = []

    for contour in contours:
        bounding_box = cv.boundingRect(contour)
        if bounding_box[2] > cutoff or bounding_box[3] > cutoff:
            bounding_boxes.append(bounding_box)

    bounding_boxes.sort()
    return bounding_boxes


def calculate_border(width, height):
    top, bottom, left, right = 0, 0, 0, 0
    if width > height:
        top = bottom = math.floor((width - height) / 2)
        if (width - height) % 2:
            bottom += 1
    else:
        left = right = math.floor((height - width) / 2)
        if (height - width) % 2:
            right += 1
    return top, bottom, left, right


def normalize(image):
    resized = cv.resize(image, (200, 200), interpolation=cv.INTER_LINEAR)
    resized = cv.resize(resized, (50, 50), interpolation=cv.INTER_LINEAR)
    (thresh, resized) = cv.threshold(resized, CONTRAST_TRESHOLD, 255, cv.THRESH_BINARY)
    return resized


if __name__ == '__main__':

    cropped_symbols = []
    print('processing ' + IMAGE_PATH)
    image = cv.imread(IMAGE_PATH)
    image = color_format_to_greyscale(image)

    cv.imshow('graycsale image', image)
    cv.waitKey(0)

    contrast_image = contrastify(image)
    cv.imshow('graycsale image', contrast_image)
    cv.waitKey(0)

    bounding_boxes = find_bounding_boxes(contrast_image)
    for i, bounding_box in enumerate(bounding_boxes):
        x, y, width, height = bounding_box
        top, bottom, left, right = calculate_border(width, height)

        cropped_image = image[y:y + height, x:x + width]
        border_image = cv.copyMakeBorder(
            cropped_image,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv.BORDER_CONSTANT,
            value=[BORDER_COLOR]
        )
        final = normalize(border_image)
        cropped_symbols.append(final)
        # cv.imshow('graycsale image', final)
        # fcv.waitKey(0)

model = NeuralNetwork().double()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
with open(CLASS_DICT_PATH, 'rb') as f:
    class_ids = pickle.load(f)
print(class_ids)

final = final / 255
final = final[None, ...]

print(final.shape)
print(torch.from_numpy(final).type)
print(torch.from_numpy(final).shape)
predic = model(torch.from_numpy(final))
predic = predic.tolist()[0]

max_value = max(predic)
max_index = predic.index(max_value)

print(class_ids.get(max_index), '=', max_value)
print(predic)
expression = ''
for symbol in cropped_symbols:
    symbol = symbol / 255
    symbol = symbol[None, ...]
    predic = model(torch.from_numpy(symbol))
    predic = predic.tolist()[0]

    max_value = max(predic)
    max_index = predic.index(max_value)
    expression += class_ids.get(max_index)

symbol_dict = {'plus': '+',
        'minus': '-',
        'div': '/',
        'mul': '*',
        'colon-open': '(',
        'colon-close': ')',
        }
for key, value in symbol_dict.items():
    expression = expression.replace(key, value, -1)

print(expression)
print(expression, '=', parse_expression(expression))
