# -*- coding: utf-8 -*-
"""
    This python code is used to detect the center of a phone laying on the floor

    Author: Linhai Li
"""

from phone_model import *
import cv2
import sys
import os
import pickle

if __name__ == "__main__":
    #Load the train SVM classifier
    model_fn = os.path.join(os.path.abspath(os.path.curdir), 'trained_classifier.dat')
    mdl = pickle.load(open(model_fn, 'rb'))

    #get the image filename
    fn = sys.argv[1]
    fn = os.path.abspath(fn)
    imgID = os.path.basename(fn) #the imgID of the test imag

    #Read image
    tempImg = cv2.imread(fn)

    #Get the sub-images as the candidate regions of interest
    candImgs, candRects = get_candidate_regions(fn, train=False)

    #Predict the probabilities of all the regions of interest
    hog_hists = []
    for img in candImgs:
        hog_hist = hog(img)
        hog_hists.append(hog_hist)

    if len(hog_hists)>0:
        hog_hists = np.float32(hog_hists)
        probs = mdl.predict_proba(hog_hists)[:,1]
        idx = np.argmax(probs)
        bbox = candRects[idx] #take the region with highest probability to be a phone
    else:
        bbox = None

    #when no suspected region is found, simply output 'The phone is not found'
    if len(candImgs)==0 or bbox is None:
        print('The phone {} is not found'.format(imgID)) #the phone is not found
    else:
        cx = bbox[0] + bbox[2]/2.0
        cy = bbox[1] + bbox[3]/2.0

        #normalized to the shape
        cx = cx/tempImg.shape[1]
        cy = cy/tempImg.shape[0]

        print('{:.4f} {:.4f}'.format(cx, cy))







