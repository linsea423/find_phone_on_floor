# -*- coding: utf-8 -*-
"""
    This module contains the essential functions for the phone model

    Author: Linhai Li
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_candidate_regions(filename, cx=0, cy=0, maxBoxes=5, LThres=0.1, HThres=0.95, train_dir=None, train=False):
    """
       Purpose: get the regions of interests for the model
       Parameters:
           filename: the filename to the original image
           cx, cy: the center of the phone; needed when training the model
           maxBoxes: maximum number of bounding boxes to generate by the edgeboxes methods; it can be tunned
           LThres, HThres: thresholds to choose pixels based on CDF of histograms of an image; can be tuned
           train: boolean; indicate if in training phase
    """
    if train:
        #Folder to save the extracted subimgs for training a phone classifier
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        #pandas dataframe to save the labels
        trainLabel_fn = os.path.join(train_dir, 'trainLabels.csv')
        if os.path.exists(trainLabel_fn):
            imgLabels = pd.read_csv(trainLabel_fn)
        else:
            imgLabels = pd.DataFrame(columns=['filename', 'label'])

    base_fn = os.path.basename(filename)
    img = cv2.imread(filename) #load data
    w,h = img.shape[1], img.shape[0]

    ######Method 1 to get candidate areas######################3
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('edgeboxes_model.yml')
    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    #Parameters to the createEdgeBoxes can be tuned
    edge_boxes = cv2.ximgproc.createEdgeBoxes(maxBoxes=maxBoxes, minBoxArea=300)
    bboxes = edge_boxes.getBoundingBoxes(edges, orimap)

    ####Method 2 to get candidate areas############
    gimg = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise
    img_gray = cv2.cvtColor(gimg, cv2.COLOR_BGR2GRAY)  #Convert to grayscale
    img_gray = cv2.equalizeHist(img_gray) #Equalize the histogram of the image to increase contrast
    mask = img_gray.copy()

    ###Attempt to identify the black screen of the phone based on histogram of the images
    hist,bins = np.histogram(img_gray, 256, [0,256]) #Compute the histogram
    hist = np.float32(hist)
    cdf = hist.cumsum()
    cdf = cdf/cdf.max()

    #take the bottom LThres values (black screen) and top HThres values (white patches)
    LcutValue = np.argwhere((cdf<=LThres))[-1]
    HcutValue = np.argwhere((cdf>=HThres))[0]

    ###Get the masks for black objects#########
    mask[(img_gray>LcutValue) & (img_gray<HcutValue)]=0
    mask[(img_gray<=LcutValue) | (img_gray>=HcutValue)]=255
    ####Use morphology techniques to remove the noise and get better mask for a phone####
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)

    #Contour detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bb = cv2.boundingRect(box)
        bb = np.asarray(cv2.boundingRect(box))
#            if bb[2]*bb[3]<300:
#                continue
        bb = np.reshape(bb, (1,4))
        bboxes = np.append(bboxes, bb, axis=0)

    #Preparation for creating subimg for training phase
    cropNum = 0

    #Preparation for prediction phase
    candImg = []
    candRect = []

    for bbox in bboxes:
        #make the region a little bit larger to include surrounding pixels
        left = np.floor(bbox[0] - bbox[2]*0.05) if (bbox[0] - bbox[2]*0.05)>=0 else 0
        top = np.floor(bbox[1] - bbox[3]*0.05) if (bbox[1] - bbox[3]*0.05)>=0 else 0
        right = np.ceil(left + bbox[2]*1.1)
        bottom = np.ceil(top + bbox[3]*1.1)
        left,right,top,bottom = int(left),int(right),int(top),int(bottom)

        mask[mask>0] = 0
        mask[top:bottom, left:right] = 255
        subImg = cv2.bitwise_and(img, img, mask=mask)

        #Get the sub-region from the original image
        subImg = subImg[top:bottom, left:right]
        subImg = cv2.resize(subImg, (45, 35))

        if train:
            subImg_fn = base_fn[:-4] + '_notP_' + str(cropNum) +'.jpg'
            phoneLabel = -1

            bx,by,bw,bh = bbox
            xp = (bx + bw/2.0)/w
            yp = (by + bh/2.0)/h

            if np.sqrt((xp-cx)**2+(yp-cy)**2)<=0.05:
                subImg_fn = base_fn[:-4] + '_Phone_' + str(cropNum) +'.jpg'
                phoneLabel = 1

            cropNum += 1

            cv2.imwrite(os.path.join(train_dir, subImg_fn), subImg)
            tempDF = pd.DataFrame([[subImg_fn, phoneLabel]], columns=['filename', 'label'])
            imgLabels = imgLabels.append(tempDF, ignore_index=True, verify_integrity=True)

        else:
            candImg.append(subImg)
            candRect.append(bbox)

    if train:
        imgLabels.to_csv(trainLabel_fn, index=False)
        pass
    else:
        return candImg, candRect

def hog(img, bin_n=32):
    """
       Purpose: compute the HOG feature of a image
       Parameters:
           img: the image array to compute HOG on
           bin_n: number of bins in the computed HOG; it can be tunned
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hog_hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hog_hist = np.hstack(hog_hists)     # hist is a 64 bit vector

    return hog_hist

def explore_img(filename, cx, cy, LThres=0.1, HThres=0.95):
    """
       Purpose: Used to explore how the thresholding and edgebox methods work on some images
       Parameters:
           filename: the filename to the original image
           cx, cy: the center of the phone; needed when training the model
           LThres, HThres: thresholds to choose pixels based on CDF of histograms of an image; can be tuned
    """
    basefile = os.path.basename(filename) #base filename of the image

    img = cv2.imread(filename) #load data
    img_edge = img.copy()
    gimg = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise
    img_gray = cv2.cvtColor(gimg, cv2.COLOR_BGR2GRAY)  #Convert to grayscale
    img_gray = cv2.equalizeHist(img_gray) #Equalize the histogram of the image to increase contrast

    hist,bins = np.histogram(img_gray, 256, [0,256]) #Compute the histogram
    hist = np.float32(hist)
    cdf = hist.cumsum()
    cdf = cdf/cdf.max()

    #take the bottom 10% values (black screen) and top 5% values (white patches)
    LcutValue = np.argwhere((cdf<=LThres))[-1]
    HcutValue = np.argwhere((cdf>=HThres))[0]

    ###Get the masks for black objects#########
    mask = img_gray.copy()
    mask[(img_gray>LcutValue) & (img_gray<HcutValue)]=0
    mask[(img_gray<=LcutValue) | (img_gray>=HcutValue)]=255
    ####Use morphology techniques to remove the noise and get better mask for a phone####
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)

    ####show the gray image and the mask###############
    res = np.hstack((img_gray, 255-mask))
    cv2.imshow(basefile+'_gray', res)

    w,h = img.shape[1], img.shape[0]
    #Contour detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bb = cv2.boundingRect(box)
#            if bb[2]*bb[3]<300:
#                continue
        bx,by,bw,bh = bb
        xp = (bx + bw/2.0)/w
        yp = (by + bh/2.0)/h

        if np.sqrt((xp-cx)**2+(yp-cy)**2)<=0.05:
            img = cv2.drawContours(img,[box],0,(0,255,0),2)
        else:
            img = cv2.drawContours(img,[box],0,(255,0,0),2)

    cv2.imshow(basefile+'_thrs_box', img)

    ######EdgeBoxes######################3
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('edgeboxes_model.yml')
    rgb_im = cv2.cvtColor(img_edge, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    #The parameters to the EdgeBoxes can be tuned
    edge_boxes = cv2.ximgproc.createEdgeBoxes(maxBoxes=5, minBoxArea=300, edgeMinMag=0.05, clusterMinMag=0.7)
    bboxes = edge_boxes.getBoundingBoxes(edges, orimap)

    cv2.imshow(basefile+'_edges', edges)
    for bbox in bboxes:
        bx,by,bw,bh = bbox
        xp = (bx + bw/2.0)/w
        yp = (by + bh/2.0)/h

        if np.sqrt((xp-cx)**2+(yp-cy)**2)<=0.05:
            img_edge = cv2.rectangle(img_edge,(bx, by), (bx+bw, by+bh), (0,255,0), 2, cv2.LINE_AA)
        else:
            img_edge = cv2.rectangle(img_edge,(bx, by), (bx+bw, by+bh), (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow(basefile+'_edge_box', img_edge)




