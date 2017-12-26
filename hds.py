# Module to load and preprocess the Oxford Hand Tracking Dataset
# Dataset page: http://www.robots.ox.ac.uk/~vgg/data/hands/
#
# Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)

import os
import cv2
import random
import numpy as np
import scipy.io as sio

#Size of the images for training
SIZE = (224,224)
#Use color or black and white images
COL = False

#Use these parameters for the color model
#SIZE = (192,192)
#COL = True

#Path to the dataset
RAW_DS_PATH = '/home/prithvi/dsets/hand_dataset/training_dataset/training_data'

#Parameters to inflate the dataset with synthetic data
H_FLIP  = True  #Include the horizontal mirror image

V_FLIP  = False #Include the vertical mirror image

HIST_EQ = False #Use the histogram equalized version of the image

BLUR = -1       #A tuple indicating the dimension of Gaussian blur kernel (E.g: (3,3))
                #Set to -1 to not use the blurred image

GAMMA = []      #List of floats indicating the values of gamma
                #Use the gamma corrected version of the images with these values

#Method to run histogram equalization on an image
def eq_hist(img):
    if COL == True:
        he = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        he[:,:,0] = cv2.equalizeHist(he[:,:,0])
        he = cv2.cvtColor(he,cv2.COLOR_YUV2BGR)
    else:
        he = cv2.equalizeHist(img)
    return he

#Method to normalize an image
def norm(img):
    #Add normalization steps here if needed
    if COL == True:
        return img
    else:
        return img[...,np.newaxis]

#Method to run gamma correction on an image
#Source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

#Method to display a list of images
def disp(img_list,name=None):
    if name is None:
        name = 'Image'
    for i,img in enumerate(img_list):
        img = np.uint8(img)
        cv2.imshow(name+' '+str(i),img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#Load data for one image
def load_sample_data(name):
    img_path = os.path.join(RAW_DS_PATH,'images')
    ant_path = os.path.join(RAW_DS_PATH,'annotations')
    if COL == True:
        img = cv2.imread(os.path.join(img_path,name+'.jpg'))
    else:
        img = cv2.imread(os.path.join(img_path,name+'.jpg'),0)
    try:
        boxes = sio.loadmat(os.path.join(ant_path,name+'.mat'))['boxes']
    except IOError:
        return None
    sdata = []
    for box in boxes[0]:
        v1 = box[0][0][0][0]
        v2 = box[0][0][1][0]
        v3 = box[0][0][2][0]
        v4 = box[0][0][3][0]
        v1 = np.array(map(int,v1)[::-1],np.int32)
        v2 = np.array(map(int,v2)[::-1],np.int32)
        v3 = np.array(map(int,v3)[::-1],np.int32)
        v4 = np.array(map(int,v4)[::-1],np.int32)
        sdata.append( np.array([v1,v2,v3,v4],np.int32) )
    return (img,sdata)

def draw_boxes(img,bbox):
    for v1,v2,v3,v4 in bbox:
        points = np.array()
        cv2.line(img,v1,v2,(0,255,0),1)
        cv2.line(img,v2,v3,(0,255,0),1)
        cv2.line(img,v3,v4,(0,255,0),1)
        cv2.line(img,v4,v1,(0,255,0),1)
    disp([img,mask])

#Method to load the dataset
def load_ds():
    ds = []

    for name in os.listdir(os.path.join(RAW_DS_PATH,'images')):
        name = name[:-4]
        sdata = load_sample_data(name)
        if sdata is None:
            continue
        img,bbox = sdata
        mask = np.zeros((img.shape[0],img.shape[1]))
        for box in bbox:
            cv2.fillConvexPoly(mask,box,255)        
        img = cv2.resize(img,SIZE)
        mask = cv2.resize(mask,SIZE)
        mask[mask>0] = 1
        
        #Use only images which have the hands covering a large area
        if np.sum(mask) > 17*17:
            #Add the image to the dataset
            ds.append((norm(img),mask[...,np.newaxis]))
            
            #Add the vertical flip 
            if V_FLIP == True:
                v = cv2.flip(img,0)
                vmask = cv2.flip(mask,0)
                ds.append((norm(v),vmask[...,np.newaxis]))
            
            #Add the horizontal flip
            if H_FLIP == True:
                h = cv2.flip(img,1)
                hmask = cv2.flip(mask,1)
                ds.append((norm(h),hmask[...,np.newaxis]))
            
            #Add blurred versions of the image
            if BLUR != -1:
                ds.append((norm(cv2.GaussianBlur(img,(3,3),0)),mask[...,np.newaxis]))
                if V_FLIP == True:
                    ds.append((norm(cv2.GaussianBlur(v,(3,3),0)),vmask[...,np.newaxis]))
                if H_FLIP == True:
                    ds.append((norm(cv2.GaussianBlur(h,(3,3),0)),hmask[...,np.newaxis]))
            
            #Add the histogram equalized version of the image
            if HIST_EQ == True:
                ds.append((norm(eq_hist(img)),mask[...,np.newaxis]))
                if V_FLIP == True:
                    ds.append((norm(eq_hist(v)),vmask[...,np.newaxis]))
                if H_FLIP == True:
                    ds.append((norm(eq_hist(h)),hmask[...,np.newaxis]))
            
            #Add the gamma corrected versions of the image
            for g in GAMMA:
                ds.append((norm(adjust_gamma(img,gamma=g)),mask[...,np.newaxis]))
                if V_FLIP == True:
                    ds.append((norm(adjust_gamma(v,gamma=g)),vmask[...,np.newaxis]))
                if H_FLIP == True:
                    ds.append((norm(adjust_gamma(h,gamma=g)),hmask[...,np.newaxis]))

    random.shuffle(ds)
    return ds
