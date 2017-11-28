# -*- coding: utf-8 -*-
"""
    This python code is used to detect the center of a phone laying on the floor
    
    Author: Linhai Li
"""

from phone_model import *
import sys
import pandas as pd
import os
import numpy as np
import cv2
import pickle

#Load the train SVM classifier
model_fn = os.path.join(os.path.abspath(os.path.curdir), 'svm_classifier.dat')
svm = pickle.load(open(model_fn, 'rb'))

#plt.close('all')
#cv2.destroyAllWindows()

#get the image filename
fn = sys.argv[1]
fn = os.path.abspath(fn)

basefile = os.path.basename(fn) #the imgID of the test imag

#Read image                           
tempImg = cv2.imread(fn)

#Create a phone_img instance and then find the suspected regions that may contain the phone
img_cl = phone_img()
candImgs, candRects = img_cl.get_candidate_regions(fn, train=False)

#when there are more than 1 suspected region
if len(candImgs)>1:
    hists = []
    for img in candImgs:
        hist = img_cl.hog(img)
        hists.append(hist)
        
    hists = np.float32(hists)
    probs = svm.predict_proba(hists)[:,1]
    idx = np.argmax(probs)
    
    bbox = candRects[idx] #take the region with highest probability to be a phone
    
#When there is only one region of interest, output the center of the region
elif len(candImgs)==1:
    bbox = candRects[0]

#when no suspected region is found, simply output a coordinate 0.5 0.5
if len(candImgs) == 0:         
    print('0.5 0.5') #the phone is not found
else:
    
    M = cv2.moments(bbox)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
#    #Display the image and detected center to check if the phone is detected correctly
#    cv2.drawMarker(tempImg, (cx, cy), (0,0,255), 1, 6, 2)
#    cv2.imshow(basefile, tempImg)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    #normalized to the shape
    cx = 1.*cx/tempImg.shape[1]
    cy = 1.*cy/tempImg.shape[0]
    
    #I am using Python 3.5, I can use much simpler syntax in python 3.5
    #I hope this works for Python 2.7. If it doesn't, please comment these two lines and uncomment the last line
    outstr = format(cx, '.4f') + ' ' + format(cy, '.4f')
    print(outstr) 
    
    #In Python 2.7, use this
    # print '%.4f %.4f' %(cx, cy)
    
    
    
    
    
    
