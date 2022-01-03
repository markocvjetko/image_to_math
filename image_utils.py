import cv2 as cv
import math


def color_format_to_greyscale(image):
    '''
    Changes the image format to greyscale.

    :param image: Image to be changed.
    :return:
    '''
    if len(image.shape) == 3:  # shape 3 for colored images, 2 for greyscale
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        return image


def contrastify(image, contrast_treshold):
    '''
    Transforms a greyscale image into pure black or white image. Each pixel is transformed
    into black or white depending its value and the contrast_treshold parameter.

    :param image:
    :param contrast_treshold:
    :return:
    '''
    (thresh, image) = cv.threshold(image, contrast_treshold, 255, cv.THRESH_BINARY)
    return image  # invert colors


def negative(image):
    '''
    Returns a negative of an image

    :param image: Image to be returned as negative.
    :return:
    '''
    return 255 - image


def zero_one_format(image):
    '''
    Transforms color values of a greyscale image from [0, 255] to [0, 1].

    :param image: Image to be transformed.
    :return:
    '''
    return image / 255


def find_bounding_boxes(image, cutoff=40):
    '''
    Returns bounding boxes of contours found on the image. Smaller bounding boxes
    are discared, specified by the cutoff treshold.

    Parameters:
        image (ndarray): Image whose bounding boxes are searched for.
        cutoff (int):Specifies which bounding boxes are discarded. If a bounding box
        has height or width smaller than cutoff, it is discarded.
    '''

    contours, _ = cv.findContours(image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    bounding_boxes = []

    for contour in contours:
        bounding_box = cv.boundingRect(contour)
        if bounding_box[2] > cutoff or bounding_box[3] > cutoff:
            bounding_boxes.append(bounding_box)

    bounding_boxes.sort()
    return bounding_boxes


def calculate_border(width, height):
    '''
    Returns top, bottom, left and right border pixel lengths to make the image
    square shaped.

    Parameters:
        width (int):Image pixel width.
        height (int):Image pixel height.
    '''

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
    image = cv.resize(image, (200, 200), interpolation=cv.INTER_LINEAR)
    image = cv.resize(image, (50, 50), interpolation=cv.INTER_LINEAR)
    return image


def show_image(image, image_title='image', wait_key=0):
    '''
    Displays an image in a window.

    :param image: Image to be displayed.
    :param image_title: Title of the image.
    :param wait_key: Display time length in seconds. If zero, image display
    closes on pressing any button.
    :return:
    '''
    cv.imshow(image_title, image)
    cv.waitKey(wait_key)


def crop_image(image, bounding_box):
    '''
    Returns the cropped image specified by the bounding box

    :param image (ndarray):
    :param bounding_box (int, int, int, int): (x, y, width, height)
    :return:
    '''
    x, y, width, height = bounding_box

    return image[y:y + height, x:x + width]

def draw_bounding_boxes(image, bounding_boxes, color=127):
    '''
    Draws specified bounding boxes on the image

    :param image: Image to draw on
    :param bounding_boxes: Bounding boxes to draw
    :param color: greyscale color of bounding boxes
    :return:
    '''
    image_bbs = image
    for x, y, width, height in bounding_boxes:
        image_bbs = cv.rectangle(image_bbs, (x, y), (x+width, y+height), color=color, thickness=2)
    return image_bbs