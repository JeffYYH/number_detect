import cv2
import numpy as np
from pynput.keyboard import Controller

def normalize_sequence(seq): #normalize the data in arraay
    if len(seq) < 2:
        return []
    xs = [p[0] for p in seq]
    ys = [p[1] for p in seq]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width == 0 or height == 0:
        return []
    # Translate to origin (0,0)
    translated = [(p[0] - min_x, p[1] - min_y) for p in seq]
    # Scale by the larger dimension to preserve aspect ratio
    scale = max(width, height)
    normalized = [(x / scale, y / scale) for x, y in translated]
    return normalized

def sequence_to_image(seq, size=28):

    img = np.zeros((size, size), dtype=np.uint8)
    normalized = normalize_sequence(seq)
    if not normalized:
        return img
    # Scale coordinates to image size
    points = [(int(x * (size - 1)), int(y * (size - 1))) for x, y in normalized]
    for i in range(1, len(points)):
        cv2.line(img, points[i - 1], points[i], 255, 1)
    return img

def output(digit):
    keyboard = Controller()
    keyboard.press(str(digit))
    keyboard.release(str(digit))

def backspace():
    keyboard = Controller()
    keyboard.press('\b')
    keyboard.release('\b')
