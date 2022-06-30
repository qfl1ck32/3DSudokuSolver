import cv2 as cv
import numpy as np

from glob import glob


def read_image(file_path: str, flags=None):
    return cv.imread(file_path, flags=flags)


def read_images(folder_path: str, limit=None, format="*", flags=None):
    return list(map(lambda img: read_image(img, flags), glob(f"{folder_path}/{format}")[:limit]))

def show_image(image: np.ndarray, window_name="-", window_size=(480, 480)):
    image = cv.resize(image, window_size)

    cv.imshow(window_name, image)

    cv.waitKey(0)

    cv.destroyWindow(window_name)

    return image


def rotate_bound(image, angle, borderValue=None):
    (h, w) = image.shape[:2]
    (cx, cy) = (w/2, h/2)

    M = cv.getRotationMatrix2D((cx, cy), -angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))

    M[0, 2] += (nW/2) - cx
    M[1, 2] += (nH/2) - cy

    return cv.warpAffine(image, M, (nW, nH), borderValue=borderValue)


def sliding_window(image: np.ndarray, step: int, window_size: tuple[int, int], eps=0):
    window_x, window_y = window_size[0], window_size[1]
    image_height, image_width = image.shape[0], image.shape[1]

    for y in range(0, image_height, step):
        for x in range(0, image_width, step):
            yield x, y, image[y: y + window_y + eps, x: x + window_x + eps]


def shift_image(image: np.array, shift_x: int, shift_y: int, borderValue=(255, 255, 255)):
    width, height = image.shape[:2]

    M = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    return cv.warpAffine(image, M, (height, width), borderMode=cv.BORDER_CONSTANT, borderValue=borderValue)


def add_text_at_point(img: np.ndarray, txt: str, point: tuple, circle_size=48, text_scale=1.5, text_thickness=2, text_face=cv.FONT_HERSHEY_DUPLEX, color=(0, 0, 255)):
    cv.circle(img, point, circle_size, color, -1)

    text_size, _ = cv.getTextSize(
        txt, text_face, text_scale, text_thickness)

    text_origin = (int(point[0] - text_size[0] / 2),
                   int(point[1] + text_size[1] / 2))

    cv.putText(img, txt, text_origin, text_face,
                text_scale, (255, 255, 255), text_thickness, cv.LINE_AA)


def get_new_coordinates_after_image_resize(original_size: np.ndarray, new_size: np.ndarray, original_coordinate):
    original_size = np.array(original_size[:2])

    new_size = np.array(new_size[:2])

    original_coordinate = np.array(original_coordinate)

    new_coordinates = original_coordinate / (original_size / new_size)

    return (int(new_coordinates[0]), int(new_coordinates[1]))


def unpad_image(image, pad_size, rgb=True):
    return image[pad_size: -pad_size, pad_size: -pad_size, :] if rgb else image[pad_size: -pad_size, pad_size: -pad_size]


def order_points(points: np.ndarray, dtype=np.int32):
    rectangle = np.zeros((4, 2), dtype=dtype)

    sum = points.sum(axis=1)

    rectangle[0] = points[np.argmin(sum)]
    rectangle[2] = points[np.argmax(sum)]

    difference = np.diff(points, axis=1)

    rectangle[1] = points[np.argmin(difference)]
    rectangle[3] = points[np.argmax(difference)]

    return rectangle


def create_structure(size: tuple[int]):
    struct = cv.getStructuringElement(cv.MORPH_RECT, size)
    struct[struct == 1] = 255

    return struct


def extract_coordinates_from_contour(contour: np.array):
    return np.array(list(map(lambda c: c[0], contour)))

def four_point_transform(image: np.ndarray, points: np.array, inverse=False):
    top_left, top_right, bottom_right, bottom_left = points

    width_bototm = np.sqrt(np.sum(np.square(bottom_right - bottom_left)))
    width_top = np.sqrt(np.sum(np.square(top_right - top_left)))
	
    max_width = max(int(width_bototm), int(width_top))

    height_bottom = np.sqrt(np.sum(np.square(top_right - bottom_right)))
    height_top = np.sqrt(np.sum(np.square(top_left - bottom_left)))
    
    max_height = max(int(height_bottom), int(height_top))
   
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype = "float32")

    M = cv.getPerspectiveTransform(dst, points) if inverse else cv.getPerspectiveTransform(points, dst)

    return cv.warpPerspective(image, M, (max_width, max_height)), M
