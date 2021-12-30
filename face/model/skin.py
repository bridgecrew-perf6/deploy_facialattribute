import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import math
import pprint
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

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
