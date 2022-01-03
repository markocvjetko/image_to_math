import os
import pickle

import torch

from NeuralNetwork import NeuralNetwork
from image_utils import *
from math_parser import parse_expression
from math_utils import symbol_words_to_symbol_characters

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = ROOT_DIR + '/image-examples/11.jpg'
MODEL_PATH = ROOT_DIR + '/model.pth'
CLASS_DICT_PATH = ROOT_DIR + '/class-ids.txt'

BORDER_COLOR = 255
CONTRAST_TRESHOLD = 90

def load_model_and_class_id_dict():
    model = NeuralNetwork().double()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with open(CLASS_DICT_PATH, 'rb') as f:
        class_ids = pickle.load(f)
    return model, class_ids


def calculate():

    cropped_symbols = []
    image = cv.imread(IMAGE_PATH)
    image = color_format_to_greyscale(image)
    show_image(image, 'image')
    contrast_image = contrastify(image, CONTRAST_TRESHOLD)
    contrast_image = negative(contrast_image)
    show_image(contrast_image, 'image')
    bounding_boxes = find_bounding_boxes(contrast_image)
    
    for i, bounding_box in enumerate(bounding_boxes):
        x, y, width, height = bounding_box
        cropped_image = crop_image(image, bounding_box)

        top, bottom, left, right = calculate_border(width, height)
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
        final = contrastify(final, CONTRAST_TRESHOLD)
        cropped_symbols.append(final)

        show_image(final, 'image')

    image_bbs = draw_bounding_boxes(contrast_image, bounding_boxes)
    show_image(image_bbs, 'image')

    model, class_ids = load_model_and_class_id_dict()
    expression = ''

    for symbol in cropped_symbols:
        symbol = zero_one_format(symbol)
        symbol = symbol[None, ...]                                      # adds a dimension (needed for the model, why?) (batches?)
        symbol_tensor = torch.from_numpy(symbol)
        prediction = model(symbol_tensor)
        prediction = prediction.tolist()[0]                             # "probability" for each class as list

        max_pred_val = max(prediction)
        max_index = prediction.index(max_pred_val)
        expression += class_ids.get(max_index)                          # append most likely class from dict
        expression = symbol_words_to_symbol_characters(expression)      # parse math operators as words to symbols

    print(expression)
    print(expression, '=', parse_expression(expression))                # math calculator call
    return str(expression) + '=' + str(parse_expression(expression))

if __name__ == '__main__':
    calculate()
