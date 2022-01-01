from django.shortcuts import render
from keras.models import load_model
import scipy.io
import pandas as pd
import keras
from keras.preprocessing import image
from keras.models import Model
from django.http import HttpResponse
from . models import Uploadimage
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
from skimage import io
import urllib.request 
import cv2
import os
from PIL import Image, ImageOps
import numpy as np
import face_recognition
import tensorflow as tf
from sklearn.cluster import KMeans
from collections import Counter
import math
import pprint
from matplotlib import pyplot as plt
from imutils import face_utils
import ctypes
import copy
from glob import glob
from colormap import rgb2hex
from colormap import hex2rgb
# Create your views here.
# Methods for Eyes Color Detection
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C
def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
def eye_segment(image):
    p = r'D:\textutils\face\data_model\shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    im = cv2.imread(image,1)
    desired_size = 500
    old_size = im.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    image = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    img1=copy.deepcopy(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    xval=[]
    yval=[]
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        i=1
        for (x, y) in shape:
            if any( [i == 38, i == 39,i == 41, i == 42,i == 44, i == 45,i == 47, i == 48]):
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                    xval.append(x)
                    yval.append(y)
            i=i+1
        val=[0,4]
        point=[]
        for i in val:
            cv2.line(image,(xval[i], yval[i]),(xval[i+2], yval[i+2]),(255,0,0),2)
            cv2.line(image,(xval[i+1], yval[i]),(xval[i+3], yval[i+2]),(255,0,0),2)
            L1 = line([xval[i], yval[i]], [xval[i+2], yval[i+2]])
            L2 = line([xval[i+1], yval[i+1]], [xval[i+3], yval[i+3]])

            R = intersection(L1, L2)
            point.append(R)
            cv2.circle(image, (int(R[0]), int(R[1])), 2, (0, 0, 255), -1)
    cropped = img1[yval[0]:yval[2],xval[0]:xval[1]]
    return cropped
def difference_points(num_list): 
    best_diff=100000
    for num in num_list:
        for num2 in num_list:
            if not(num==num2):
                diff=np.abs(num-num2)
                if diff<best_diff:
                    points=[num,num2]
                    best_diff=diff
    return points
def im_components(labels,stats,method=1):
    temp=[]
    for i in range(1,stats.shape[0]):
        width=stats[i,2]
        height=stats[i,3]
        if method==2:  
            if (height/stats[0,3] > 0.15 ) or  (height/stats[0,3] < 0.05) or (width/stats[0,2] > 0.4 ) or (width/stats[0,2] < 0.2 ):
                labels[labels==i]=0
            else:
                temp.append(stats[i,1]+stats[i,3]/2)
        else:
            temp.append(stats[i,1]+stats[i,3]/2)
    new_stats=[]
    closest_points=difference_points(temp)  
    for i in range(1,stats.shape[0]):
        width=stats[i,2]
        height=stats[i,3]
        temp_stat=stats[i,1]+stats[i,3]/2
        if not(temp_stat in closest_points):
            labels[labels==i]=0
        else:
            new_stats.append(stats[i])
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    return np.array(new_stats),labeled_img,temp
def compute_eyecolor(orig_img):
    method=2
    img_patch = eye_segment(orig_img)
    hsv_img_patch=cv2.cvtColor(img_patch, cv2.COLOR_BGR2HSV)
    limit=[]
    for i in range(3):
        temp_list=hsv_img_patch[:,:,i].reshape(-1)
        temp_list=temp_list[temp_list!=0]
        rand_indices=np.unique(np.random.randint(temp_list.shape,size=temp_list.shape)[:500])
        temp_list=temp_list[rand_indices]
        limit.append(s.mode(temp_list)[0][0])
    eye_color=np.zeros(hsv_img_patch.shape).astype(np.uint8)
    eye_color[:,:,0]=int(limit[0])
    eye_color[:,:,1]=int(limit[1])
    eye_color[:,:,2]=int(limit[2])
    eye_color=cv2.cvtColor(eye_color, cv2.COLOR_HSV2RGB)
    HEX_color='#%02x%02x%02x' % (eye_color[0,0,0], eye_color[0,0,1], eye_color[0,0,2])
    eye_color_name = color_name(HEX_color)
    color_list = ['Black','Blue','Brown','Grey']
    return color_list[eye_color_name]
def color_name(hex):
    point1 = [np.array((0, 0, 0)),np.array((0, 0, 255)),np.array((165, 42, 42)),np.array((50.2, 50.2, 50.2))]
    point2 = np.array(hex2rgb(str(hex)))
    label_name = []
    for i in point1:
        label_name.append(np.linalg.norm(i - point2))
    return label_name.index(min(label_name))
# Methods for Face Detection
def Face_Cropped(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'D:\textutils\face\data_model\shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    filename_path = image
    image = cv2.imread(image)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)
        cv2.imwrite(filename_path , faceAligned)
    return True
# Method for Front Face Check
def good_picture_check(p, debug=False):
    ## To scale for picture size
    width_im = (p[16][0] - p[0][0]) / 100
    
    ## Difference in height between eyes
    eye_y_l = (p[37][1] + p[41][1]) / 2.0
    eye_y_r = (p[44][1] + p[46][1]) / 2.0
    eye_dif = (eye_y_r - eye_y_l) / width_im
    
    ## Difference top / bottom point nose 
    nose_dif = (p[30][0] - p[27][0]) / width_im
    
    ## Space between face-edge to eye, left vs. right
    left_space = p[36][0] - p[0][0]
    right_space = p[16][0] - p[45][0]
    space_ratio = left_space / right_space
    
    if debug:
        print(eye_dif, nose_dif, space_ratio)
    
    ## These rules are not perfect, determined by trying a bunch of "bad" pictures
    if eye_dif > 5 or nose_dif > 3.5 or space_ratio > 3:
        return False
    else:
        return True

def Front_Face_Check(image):
    image = Image.open(image)
    image = np.asarray(image)
    landmarks = face_recognition.api._raw_face_landmarks(image)
    landmarks_as_tuples = [(p.x, p.y) for p in landmarks[0].parts()]
    if good_picture_check(landmarks_as_tuples):
        return True
    else:
        return False
# Method for gender detection
def Gender(image):
    image = Image.open(image)
    model = tf.keras.models.load_model(r'D:\textutils\face\data_model\gender.h5',compile=False)
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
        value = 'Male'
    elif np.argmax(prediction) == 1:
        value = 'Female'
    tf.function(experimental_relax_shapes=True,experimental_compile=True)
    return value
# Method for skin detection
def extractSkin(image):
    
  # Taking a copy of the image
    img =  image.copy()
  # Converting from BGR Colours Space to HSV
    img =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  
  # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
  
  # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img,lower_threshold,upper_threshold)
  
  # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
  
  # Extracting skin from the threshold mask
    skin  =  cv2.bitwise_and(img,img,mask=skinMask)
  
  # Return the Skin image
    return cv2.cvtColor(skin,cv2.COLOR_HSV2BGR)

def removeBlack(estimator_labels, estimator_cluster):
    
  # Check for black
    hasBlack = False
  
  # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

  
  # Quick lambda function to compare to lists
    compare = lambda x, y: Counter(x) == Counter(y)
   
  # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):
        color = [int(i) for i in estimator_cluster[x[0]].tolist() ]
        if compare(color , [0,0,0]) == True:
            del occurance_counter[x[0]]
      # remove the cluster 
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster,x[0],0)
            break
      
   
    return (occurance_counter,estimator_cluster,hasBlack)

def getColorInformation(estimator_labels, estimator_cluster,hasThresholding=False):
  
  # Variable to keep count of the occurance of each color predicted
    occurance_counter = None
  
  # Output list variable to return
    colorInformation = []
  
  
  #Check for Black
    hasBlack =False
  
  # If a mask has be applied, remove th black
    if hasThresholding == True:
    
        (occurance,cluster,black) = removeBlack(estimator_labels,estimator_cluster)
        occurance_counter =  occurance
        estimator_cluster = cluster
        hasBlack = black
    
    else:
        occurance_counter = Counter(estimator_labels)
 
  # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values()) 
  
 
  # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):
    
        index = (int(x[0]))
    
    # Quick fix for index out of bound when there is no threshold
        index =  (index-1) if ((hasThresholding & hasBlack)& (int(index) !=0)) else index
    
    # Get the color number into a list
        color = estimator_cluster[index].tolist()
    
    # Get the percentage of each color
        color_percentage= (x[1]/totalOccurance)
    
    #make the dictionay of the information
        colorInfo = {"cluster_index":index , "color": color , "color_percentage" : color_percentage }
    
    # Add the dictionary to the list
        colorInformation.append(colorInfo)
    
      
        return colorInformation

def extractDominantColor(image,number_of_colors=5,hasThresholding=True):
  
  # Quick Fix Increase cluster counter to neglect the black(Read Article) 
    if hasThresholding == True:
        number_of_colors +=1
  
  # Taking Copy of the image
    img = image.copy()
  
  # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  
  # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]) , 3)
  
  #Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)
  
  # Fit the image
    estimator.fit(img)
  
  # Get Colour Information
    colorInformation = getColorInformation(estimator.labels_,estimator.cluster_centers_,hasThresholding)
    return colorInformation

def plotColorBar(colorInformation):
  #Create a 500x100 black image
    color_bar = np.zeros((100,500,3), dtype="uint8")
  
    top_x = 0
    for x in colorInformation:    
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int,(x['color'])))
  
        cv2.rectangle(color_bar , (int(top_x),0) , (int(bottom_x),color_bar.shape[0]) ,color , -1)
        top_x = bottom_x
    return color_bar

def prety_print_data(color_info,color_list):
    for x in color_info:
        tlist = []
        temp = str(pprint.pformat(x)).split(":")
        t1 = temp[2].replace("[","")
        t1 = t1.replace("]","")
        t1 = t1.split(",")
        #print(t1[0]+" "+t1[1]+t1[2])
        tlist.append(float(t1[0].replace(" ","")))
        tlist.append(float(t1[1].replace(" ","")))
        tlist.append(float(t1[2].replace(" ","")))
        color_list.append(tlist)
    return color_list

def Skin(image):
    Clusters = [[106.41521431,68.94741393,51.95100356],[197.01916088,144.91564491,119.52954836],[231.93510238,185.26028206,161.16835255],[158.94689506,109.61798286,86.12439095]]
    color_list = []
    image = Image.open(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    skin = extractSkin(image)
    dominantColors = extractDominantColor(skin,hasThresholding=True)
    color_list = prety_print_data(dominantColors,color_list)
    colour_bar = plotColorBar(dominantColors)
    plt.axis("off")
    temp = []
    for c in Clusters:
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(c, color_list[0])]))
        temp.append(distance)
    if temp.index(min(temp)) == 0:
        return "Black Brown"
    elif temp.index(min(temp)) == 1:
        return "Olive"
    elif temp.index(min(temp)) == 2:
        return "Fair"
    elif temp.index(min(temp)) == 3:
        return "Brown"
# Method for Age
def loadImage(filepath):
    test_img = image.load_img(filepath, target_size=(48, 48))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img /= 255
    return test_img
def Age(image):
    age_model = load_model(r"D:\textutils\face\data_model\Age_sex_detection.h5")
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
# Method for Shape Detection
def Shape(image):
    image = Image.open(image)
    model = tf.keras.models.load_model(r'D:\textutils\face\data_model\shape.h5',compile=False)
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
        value = 'Square'
    elif np.argmax(prediction) == 1:
        value = 'Round'
    elif np.argmax(prediction) == 2:
        value = 'Oval'
    elif np.argmax(prediction) == 3:
        value = 'Diamond'
    tf.function(experimental_relax_shapes=True,experimental_compile=True)
    return value
# Method for Forehead detection
def Forehead(image):
    image = Image.open(image)
    model = tf.keras.models.load_model(r'D:\textutils\face\data_model\forehead.h5',compile=False)
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
# Method for Eyebrows length
def Eyebrows(image):
    image = Image.open(image)
    predictor_path = r"D:\textutils\face\data_model\eyebrows_predictor.dat"
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
# Method for Nose Width
def show_box(image, corners):
    pil_image = Image.fromarray(image)
    w, h = pil_image.size
    
    ## Automatically determine width of the line depending on size of picture
    line_width = math.ceil(h / 100)
    
    d = ImageDraw.Draw(pil_image) 
    d.line([corners['bottom_left'], corners['top_left']], width = line_width)
    d.line([corners['bottom_left'], corners['bottom_right']], width = line_width)
    d.line([corners['top_left'], corners['top_right']], width = line_width)
    d.line([corners['top_right'], corners['bottom_right']], width = line_width)
def get_face_points1(points, method='average', top='eyebrow'):
    width_left, width_right = points[60], points[64]
    
    if top == 'eyebrow':
        top_left = points[18]
        top_right = points[25]
        
    elif top == 'eyelid':
        top_left = points[1]
        top_right = points[15] 
        
    else:
        raise ValueError('Invalid top point, use either "eyebrow" or "eyelid"')
        
    bottom_left, bottom_right = points[33], points[35]
    
    if method == 'left':
        coords = (width_left[0], width_right[0], top_left[1], bottom_left[1])
        
    elif method == 'right':
        coords = (width_left[0], width_right[0], top_right[1], bottom_right[1])
        
    else:
        top_average = int((top_left[1] + top_right[1]) / 2)
        bottom_average = int((bottom_left[1] + bottom_right[1]) / 2)
        coords = (width_left[0], width_right[0], top_average, bottom_average)
        
    ## Move the line just a little above the top of the eye to the eyelid    
    if top == 'eyelid':
        coords = (coords[0], coords[1], coords[2] - 4, coords[3])
        
    return {'top_left' : (coords[0], coords[2]),
            'bottom_left' : (coords[0], coords[3]),
            'top_right' : (coords[1], coords[2]),
            'bottom_right' : (coords[1], coords[3])
           }
def FWHR_calc1(corners):
    width = corners['top_right'][0] - corners['top_left'][0]
    height = corners['bottom_left'][1] - corners['top_left'][1]
    return float(width)
def get_fwhr1(image_path, url=False, show=True, method='average', top='eyelid'):
    image = np.asarray(image_path)
 
    landmarks = face_recognition.api._raw_face_landmarks(image)
    landmarks_as_tuples = [(p.x, p.y) for p in landmarks[0].parts()]
    
    if good_picture_check(landmarks_as_tuples): 
        corners = get_face_points1(landmarks_as_tuples, method=method, top = top)
        fwh_ratio = FWHR_calc1(corners)

        if fwh_ratio >= 62:
            value = 'Wide'
        else:
            value = 'Narrow'
        return value
    else:
        pass

def Nose_Width(image):
    image = Image.open(image)
    guy_url = np.asarray(image)
    return get_fwhr1(guy_url, url=True, top = 'eyelid', show=False)
# Method for Nose length
def get_face_points(points, method='average', top='eyebrow'):
    width_left, width_right = points[0], points[16]
    
    if top == 'eyebrow':
        top_left = points[18]
        top_right = points[25]
        
    elif top == 'eyelid':
        top_left = points[37]
        top_right = points[43] 
        
    else:
        raise ValueError('Invalid top point, use either "eyebrow" or "eyelid"')
        
    bottom_left, bottom_right = points[33], points[35]
    
    if method == 'left':
        coords = (width_left[0], width_right[0], top_left[1], bottom_left[1])
        
    elif method == 'right':
        coords = (width_left[0], width_right[0], top_right[1], bottom_right[1])
        
    else:
        top_average = int((top_left[1] + top_right[1]) / 2)
        bottom_average = int((bottom_left[1] + bottom_right[1]) / 2)
        coords = (width_left[0], width_right[0], top_average, bottom_average)
        
    ## Move the line just a little above the top of the eye to the eyelid    
    if top == 'eyelid':
        coords = (coords[0], coords[1], coords[2] - 4, coords[3])
        
    return {'top_left' : (coords[0], coords[2]),
            'bottom_left' : (coords[0], coords[3]),
            'top_right' : (coords[1], coords[2]),
            'bottom_right' : (coords[1], coords[3])
           }
def FWHR_calc(corners):
    width = corners['top_right'][0] - corners['top_left'][0]
    height = corners['bottom_left'][1] - corners['top_left'][1]
    return float(height)
def get_fwhr(image_path, url=False, show=True, method='average', top='eyelid'):
    image = np.asarray(image_path)
 
    landmarks = face_recognition.api._raw_face_landmarks(image)
    landmarks_as_tuples = [(p.x, p.y) for p in landmarks[0].parts()]
    
    if good_picture_check(landmarks_as_tuples): 
        corners = get_face_points(landmarks_as_tuples, method=method, top = top)
        fwh_ratio = FWHR_calc(corners)

        if fwh_ratio <= 60:
            value = 'Short'
        else:
            value = 'Long'
        return value
    else:
        pass

def Nose_Length(image):
    image = Image.open(image)
    guy_url = np.asarray(image)
    return get_fwhr(guy_url, url=True, top = 'eyelid', show=False)
# Method for Chin detection
def Chin(image):
    tf.function(experimental_relax_shapes=True)
    image = Image.open(image)
    model = tf.keras.models.load_model(r'D:\textutils\face\data_model\chin.h5',compile=False)
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
        value = 'Double'
    elif np.argmax(prediction) == 1:
        value = 'Round'
    elif np.argmax(prediction) == 2:
        value = 'Square'
    tf.function(experimental_relax_shapes=True,experimental_compile=True)
    return value
# Method for mole detection
def Mole(image):
    tf.function(experimental_relax_shapes=True)
    image = Image.open(image)
    model = tf.keras.models.load_model(r'D:\textutils\face\data_model\mole.h5',compile=False)
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
        value = 'Yes'
    elif np.argmax(prediction) == 1:
        value = 'No'
    tf.function(experimental_relax_shapes=True,experimental_compile=True)
    return value
# Method for Scar detction
def Scar(image):
    tf.function(experimental_relax_shapes=True)
    image = Image.open(image)
    model = tf.keras.models.load_model(r'D:\textutils\face\data_model\scar.h5',compile=False)
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
        value = 'Yes'
    elif np.argmax(prediction) == 1:
        value = 'No'
    tf.function(experimental_relax_shapes=True,experimental_compile=True)
    return value
# Method for Moustache detection
def Moustaches(image):
    tf.function(experimental_relax_shapes=True)
    image = Image.open(image)
    model = tf.keras.models.load_model(r'D:\textutils\face\data_model\moustache.h5',compile=False)
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
        value = 'Handlebar'
    elif np.argmax(prediction) == 1:
        value = 'Horseshoe'
    elif np.argmax(prediction) == 2:
        value = 'No'
    elif np.argmax(prediction) == 3:
        value = 'Original'
    tf.function(experimental_relax_shapes=True,experimental_compile=True)
    return value
# Method for Beard Detection
def Beard(image):
    tf.function(experimental_relax_shapes=True)
    image = Image.open(image)
    model = tf.keras.models.load_model(r'D:\textutils\face\data_model\beard.h5',compile=False)
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
        value = 'The Patchy Full-Faced'
    elif np.argmax(prediction) == 1:
        value = 'French'
    elif np.argmax(prediction) == 2:
        value = 'No'
    tf.function(experimental_relax_shapes=True,experimental_compile=True)
    return value
# Method for main index page
def index(request):
    try:
        return render(request, 'index.html')
    except:
        error = "System is Responding"
        return render(request, 'error.html', {'error':error})

def result(request):
    file_path = r'D:\textutils\media'+'\\'
    if request.method == "POST":
        files = request.FILES.getlist("img")
        for filename in files:
            Uploadimage(uploadimagefile=filename).save()
            try:
                Face_Cropped(file_path+str(filename))
                try:
                    Front_Face_Check(file_path+str(filename))
                    prediction_gender_value = Gender(file_path+str(filename))
                    skin_color_value = Skin(file_path+str(filename))
                    prediction_age_value = Age(file_path+str(filename))
                    prediction_shape_value = Shape(file_path+str(filename))
                    prediction_forehead_value = Forehead(file_path+str(filename))
                    eyebrows_length_value = Eyebrows(file_path+str(filename))
                    prediction_eyes_color_value = compute_eyecolor(file_path+str(filename))
                    nose_length_value = Nose_Length(file_path+str(filename))
                    nose_width_value = Nose_Width(file_path+str(filename))
                    prediction_chin_value = Chin(file_path+str(filename))
                    prediction_mole_value = Mole(file_path+str(filename))
                    prediction_scar_value = Scar(file_path+str(filename))
                    prediction_moustache_value = Moustaches(file_path+str(filename))
                    prediction_beard_value = Beard(file_path+str(filename))
                    data_save = Uploadimage(gender=prediction_gender_value,skin=skin_color_value,age=prediction_age_value,shape=prediction_shape_value,forehead=prediction_forehead_value,eyebrows=eyebrows_length_value,eye_color=prediction_eyes_color_value,nose_length=nose_length_value,nose_width=nose_width_value,chin=prediction_chin_value,mole=prediction_mole_value,scar=prediction_scar_value,moustaches=prediction_moustache_value,beard=prediction_beard_value,filenames=str(filename))
                    data_save.save()
                except:
                    error = "Face is not Front side"
                    return render(request, 'error.html', {'error':error})
            except:
                error = "Face is not Present in Image"
                return render(request, 'error.html', {'error':error})
            break
        return render(request, 'result.html', {'filename':filename,'prediction_gender_value':prediction_gender_value,'skin_color_value':skin_color_value,'prediction_age_value':prediction_age_value,'prediction_shape_value':prediction_shape_value,'prediction_forehead_value':prediction_forehead_value,'eyebrows_length_value':eyebrows_length_value,'nose_width_value':nose_width_value,'nose_length_value':nose_length_value,'prediction_chin_value':prediction_chin_value,'prediction_mole_value':prediction_mole_value,'prediction_scar_value':prediction_scar_value,'prediction_moustache_value':prediction_moustache_value,'prediction_beard_value':prediction_beard_value,'prediction_eyes_color_value':prediction_eyes_color_value})

def error(request):
    return render(request, 'index.html')
def automatic_tagging(request):
    return render(request, 'automatic_tagging.html')
def suspect_retrieval(request):
    return render(request, 'suspect_retrieval.html')
def compare_value(one,two):
    if one == two:
        return 1
    else:
        return 0
def suspect_retrieval_result(request):
    if request.method == "POST":
        gender_value = request.POST['gender']
        skin_value = request.POST['skin']
        age_value = request.POST['age']
        shape_value = request.POST['shape']
        forehead_value = request.POST['forehead']
        eyebrows_value = request.POST['eyebrows']
        eyes_value = request.POST['eyes']
        noselength_value = request.POST['noselength']
        nosewidth_value = request.POST['nosewidth']
        chin_value = request.POST['chin']
        mark_value = request.POST['mark']
        moustache_value = request.POST['moustache']
        beard_value = request.POST['beard']
        compare = {}
        mole_value = ""
        scar_value = ""
        if mark_value == "Mole":
            mole_value = "Yes"
        else:
            mole_value = "No"
        if mark_value == "Scar":
            scar_value = "Yes"
        else:
            scar_value = "No"
        if mark_value == "Mole and Scar":
            mole_value = "Yes"
            scar_value = "Yes"
    retrieval_result = Uploadimage.objects.all()
    for result_value in retrieval_result:
        compare[result_value.filenames] = sum([compare_value(gender_value,result_value.gender)*100,compare_value(skin_value,result_value.skin)*80,compare_value(age_value,result_value.age)*90,compare_value(shape_value,result_value.shape)*60,compare_value(forehead_value,result_value.forehead)*70,compare_value(eyebrows_value,result_value.eyebrows)*60,compare_value(eyes_value,result_value.eye_color)*75,compare_value(noselength_value,result_value.nose_length)*70,compare_value(nosewidth_value,result_value.nose_width)*60,compare_value(chin_value,result_value.chin)*40,compare_value(mole_value,result_value.mole)*70,compare_value(scar_value,result_value.scar)*70,compare_value(moustache_value,result_value.moustaches)*90,compare_value(beard_value,result_value.beard)*80])
    sort_orders = sorted(compare.items(), key=lambda x: x[1], reverse=True)
    list_val = [f_val[0] for f_val in sort_orders]
    return render(request, 'suspect_retrieval_result.html',{'list_val':list_val[0:10]})

def tryanotherone(request):
    return render(request, 'automatic_tagging.html')
