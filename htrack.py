# Module to track hand movement from video camera input.
# 
# Usage:
#       python htrack.py
# or,
#       python htrack.py ./output.avi (save output to output.avi)
#
# Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)

import os
import sys
import cv2
import numpy as np
from keras.models import load_model

#Import module to load the dataset
#This will only be useful for: (1) parameters SIZE, COL and (2) the norm() function to normalize the input frame
#See hds.py
import hds

#Parameters
model = load_model('bw_model.h5')   #Path to model file
#model = load_model('col_model.h5')   #Path to model file
save_mode = False               #True if the user provides a path to save the tracking mask
out_size = (600,300)            #Size of frame for saving to file
thresh = 0.3                    #Confidence threshold to include for the mask

#Turn on save_mode if the user has given a path to save the model
if len(sys.argv) > 1:
    save_mode = True
    out_file = sys.argv[1]
    fourcc = cv2.cv.CV_FOURCC(*"DIVX")
    out = cv2.VideoWriter(out_file,fourcc,25,out_size)  #NOTE: Vide saved at 25 FPS by default. May need to be changed

#Start reading video input
vc = cv2.VideoCapture(0)
if vc.isOpened():
    rval,frame = vc.read()
else:
    rval = False

while rval:
    if hds.COL == False:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame,hds.SIZE)
    frame = hds.norm(frame)
    mask = model.predict(np.array([frame]))[0]
    mask[mask>thresh] = 255
    mask[mask<255] = 0
    cv2.imshow("input",frame)
    cv2.imshow("mask",mask)
    
    #Save video to file
    if save_mode == True:
        #Make and format the output frame
        if hds.COL == True:
            mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            vid_frame = np.uint8(np.concatenate([frame,mask],axis=1))
        else:
            vid_frame = np.uint8(np.hstack([frame,mask]))
        vid_frame = cv2.resize(vid_frame,out_size)
        vid_frame = cv2.copyMakeBorder(vid_frame, top=100, bottom=100, left=50, right=50, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
        vid_frame = cv2.resize(vid_frame,out_size)
        if hds.COL == False:
            vid_frame = cv2.cvtColor(vid_frame,cv2.COLOR_GRAY2BGR)
        cv2.imshow("vid",vid_frame)
        out.write(vid_frame)
    
    #Read new frame
    rval,frame = vc.read()
    
    #Exit on escape
    key = cv2.waitKey(10)
    if key == 27:
        break

vc.release()
out.release()
cv2.destroyAllWindows()
