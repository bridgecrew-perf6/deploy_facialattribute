import dlib
import numpy as np
from skimage import io
import math
import os
from PIL import Image, ImageOps


def Eyebrows(image):
    image = Image.open(image)
    predictor_path = r"D:\Thesis\flask2\templates\model\eyebrows_predictor.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    img = np.asarray(image)
    dets = detector(img)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
    vec = np.empty([10, 2], dtype = int)
    for b in range(10):
        vec[b][0] = shape.part(b).x
        vec[b][1] = shape.part(b).y
    x = vec[0].tolist()
    y = vec[-1].tolist()
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

    if distance <= 136:
        value = 'Short'
    else:
        value = 'Long'
    return value
