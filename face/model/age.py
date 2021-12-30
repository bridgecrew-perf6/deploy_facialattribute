from keras.models import load_model
import scipy.io
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Model

def loadImage(filepath):
    test_img = image.load_img(filepath, target_size=(48, 48))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img /= 255
    return test_img
def Age(image):
    age_model = load_model(r"D:\Thesis\flask2\templates\model\Age_sex_detection.h5")
    prediction = age_model.predict(loadImage(image))
    predict = str(np.round(prediction[1][0]))
    predict = predict.replace("[",'')
    predict = predict.replace("]",'')
    predict = predict.replace(".",'')
    predict = int(predict)
    if predict <= 17:
        value = 'Teenager'
    elif predict <= 35:
        value = 'Young Adult'
    elif predict <= 60:
        value = 'Middle Aged'
    else:
        value = 'Old Aged'
    return value
