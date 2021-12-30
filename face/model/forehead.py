import tensorflow as tf
import os
import cv2
from PIL import Image, ImageOps
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def Forehead(image):
    image = Image.open(image)
    model = tf.keras.models.load_model(r'D:\Thesis\flask2\templates\model\forehead.h5',compile=False)
    np.set_printoptions(suppress=True)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    normalized_image_array = tf.convert_to_tensor(normalized_image_array, dtype=tf.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    if np.argmax(prediction) == 0:
        value = 'Bald'
    elif np.argmax(prediction) == 1:
        value = 'Broad'
    elif np.argmax(prediction) == 2:
        value = 'Narrow'
    tf.function(experimental_relax_shapes=True,experimental_compile=True)
    return value
