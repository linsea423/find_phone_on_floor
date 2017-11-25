# -*- coding: utf-8 -*-
"""
    This module is used to preprocess the image and generate region of interests for training and testing using OpenCV
    
    Author: Linhai Li
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class phone_img:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        
    def get_candidate_regions(self, filename, train=False): #get the candidate regions
        base_fn = os.path.basename(filename)
        
        img = cv2.imread(filename) #load data   
        #print(img.shape)
        gimg = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise
        img_gray = cv2.cvtColor(gimg, cv2.COLOR_BGR2GRAY)  #Convert to grayscale
        #img_gray = cv2.equalizeHist(img_gray) #Equalize the histogram of the image to increase contrast
        mask = img_gray.copy()
        
        ###Attempt to identify the black screen of the phone based on histogram of the images
        hist,bins = np.histogram(img_gray, 256, [0,256]) #Compute the histogram
        cdf = hist.cumsum()
        cdf = cdf/cdf.max()
         
        #take the bottom 5% values
        cutValue = np.argwhere((cdf<=0.05))[-1]
       
        ###Get the masks for black objects######### 
        mask[img_gray>cutValue]=0
        mask[img_gray<=cutValue]=255
        ####Use morphology techniques to remove the noise and get better mask for a phone####
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)
        
        xloc = self.x*img.shape[1]
        yloc = self.y*img.shape[0]
        
        #Preparation for creating subimg for training phase
        cropNum = 0

        if train:
            #Folder to save the extracted subimgs for training a phone classifier
            if not os.path.exists('train_img'):
                os.mkdir('train_img')
            #pandas dataframe to save the labels 
            trainLabel_fn = os.path.join(os.path.abspath(os.path.curdir), 'train_img', 'trainLabels.csv')              
            if os.path.exists(trainLabel_fn):
                imgLabels = pd.read_csv(trainLabel_fn) 
            else:
                imgLabels = pd.DataFrame(columns=['filename', 'label'])      
                
        #Preparation for prediction phase
        candImg = []
        candRect = []
        
        #Contour detection
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
#            img = cv2.drawContours(img,[box],0,(0,0,255),2)

            length = max(rect[1])
            width = min(rect[1])
            if length*width<250: #the box is too small, it won't be a candidate
                continue

            tempImg = img_gray.copy()
#            tempMask = np.zeros(tempImg.shape, np.uint8)
#            tempMask = cv2.fillPoly(tempMask, [box], 255)
#            tempImg[tempMask==0] = 255
                        
            if train:                
                #A phone is a rectangle with certain dimensions
                if 10<=length and length<=60 and 5<=width and width<=40 \
                    and 1.1<=length/width and length/width<=2.9 and 250<=length*width and length*width<=2400:
                        
                    bbox = cv2.boundingRect(box)
                    #make the region a little bit larger to include surrounding pixels
                    left = int(bbox[0] - bbox[2]*0.05) if (bbox[0] - bbox[2]*0.05)>=0 else 0
                    top = int(bbox[1] - bbox[3]*0.05) if (bbox[1] - bbox[3]*0.05)>=0 else 0
                    right = int(left + bbox[2]*1.1)
                    bottom = int(top + bbox[3]*1.1)
                    
                    #Get the sub-region from the original image
                    subImg = tempImg[top:bottom, left:right]
                    subImg = cv2.resize(subImg, (50,50))
                    
                    subImg_fn = base_fn[:-4] + '_notP_' + str(cropNum) +'.jpg'
                    phoneLabel = 0
                    
                    #the center of a contour block
                    M = cv2.moments(box)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    if abs(cx-xloc)<=20 and abs(cy-yloc)<=20:  
                        subImg_fn = base_fn[:-4] + '_Phone_' + str(cropNum) +'.jpg'
                        phoneLabel = 1
                        
                    cropNum += 1
                    
                    # cv2.imshow(os.path.basename(subImg_fn), subImg)
                
                    cv2.imwrite(os.path.join(os.path.abspath(os.path.curdir), 'train_img', subImg_fn), subImg)
                    tempDF = pd.DataFrame([[subImg_fn, phoneLabel]], columns=['filename', 'label'])
                    imgLabels = imgLabels.append(tempDF, ignore_index=True, verify_integrity=True)        
                
                
            else:   
                #A phone is a rectangle with certain dimensions                
                if 10<=length and length<=60 and 5<=width and width<=40 \
                    and 1.1<=length/width and length/width<=2.9 and 250<=length*width and length*width<=2400:
                        
                    bbox = cv2.boundingRect(contours[i])
                    #make the region a little bit larger to include surrounding pixels
                    left = int(bbox[0] - bbox[2]*0.05) if (bbox[0] - bbox[2]*0.05)>=0 else 0
                    top = int(bbox[1] - bbox[3]*0.05) if (bbox[1] - bbox[3]*0.05)>=0 else 0
                    right = int(left + bbox[2]*1.1)
                    bottom = int(top + bbox[3]*1.1)
                    
                    #Get the sub-region from the original image
                    subImg = tempImg[top:bottom, left:right]
                    subImg = cv2.resize(subImg, (50,50))
                    
                    candImg.append(subImg)
                    candRect.append(box)

                        
        if train:
            imgLabels.to_csv(trainLabel_fn, index=False)
            pass
        else:
            return candImg, candRect  
        
    def hog(self, img, bin_n=16): #compute the HOG feature of a image
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)     # hist is a 64 bit vector
        
        return hist
        
    def img_proc(self, filename):
        """
         This method is used to explore the data and initial ideas
         i.e., process the image to check initial regions of interests             
        """
        basefile = os.path.basename(filename) #base filename of the image
        
        img = cv2.imread(filename) #load data   
        img = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #Convert to grayscale
#        img2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,21,2)
        # img2 = cv2.equalizeHist(img2) #Equalize the histogram of the image to increase contrast
        
        ###Attempt to identify the black screen of the phone based on histogram of the images
        hist,bins = np.histogram(img2, 256, [0,256]) #Compute the histogram
        cdf = hist.cumsum()
        cdf = cdf/cdf.max()
                
#        ###Try to deal with glares###
#        im = np.zeros(img2.shape, dtype=np.uint8)
#        highValue = np.argwhere(cdf>=0.98)[0]
#        if highValue>225:
#            im[img2<highValue] = 0
#            im[img2>=highValue] = 255
#        #Morphology technique to get the patch of glare area
#        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#        im = cv2.morphologyEx(im, cv2.MORPH_DILATE, kernel)
#        img4 = cv2.inpaint(img2, im, 10, cv2.INPAINT_TELEA)
#        res = np.hstack((img2,img4, im))
#        cv2.imshow(basefile, res)
        
        #take the bottom 5% values
        cutValue = np.argwhere((cdf<=0.05))[-1]
        
        ####Check the histograms############### 
        img3 = img2.copy()
        img3[img2>cutValue] = 255
        res = np.hstack((img2, img3))
        cv2.imshow(basefile+'gray', res)
        plt.figure(num=basefile)
        plt.plot(np.arange(0,256), hist, 'k-', label='Hist')
        cdf = cdf*hist.max()
        plt.plot(np.arange(0,256), cdf, 'r-', label='cdf')
        plt.vlines(cutValue, hist.min(), hist.max(), color='k', linestyle='--')
        plt.legend()
        plt.xlabel('Grayscale Intensity')
        plt.ylabel('Histogram or CDF')
        plt.show()
#       
        ###Get the masks for black objects######### 
        img3 = img2.copy()
        img3[img2>cutValue]=0
        img3[img2<=cutValue]=255
        ####Use morphology techniques to remove the noise and get better mask for a phone####
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        img3 = cv2.morphologyEx(img3, cv2.MORPH_ERODE, kernel1)
        img3 = cv2.morphologyEx(img3, cv2.MORPH_ERODE, kernel2)
        img3 = cv2.morphologyEx(img3, cv2.MORPH_DILATE, kernel1)
        img3 = cv2.morphologyEx(img3, cv2.MORPH_DILATE, kernel2)
        img3 = cv2.morphologyEx(img3, cv2.MORPH_DILATE, kernel2)
        
        xloc = self.x*img.shape[1]
        yloc = self.y*img.shape[0]
        
        #Contour detection
        _, contours, _ = cv2.findContours(img3, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
#            img = cv2.drawContours(img,[box],0,(0,0,255),2)
            
            img = cv2.drawContours(img,[box],0,(0,0,255),2)    
            #A phone is a rectangle with certain dimensions
            length = max(rect[1])
            width = min(rect[1])
            if 10<=length and length<=60 and 5<=width and width<=40 \
                    and 1.1<=length/width and length/width<=2.9 and 250<=length*width and length*width<=2400:
                        
                img = cv2.drawContours(img,[box],0,(255,0,0),2)
                
                M = cv2.moments(box)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                if abs(cx-xloc)<=20 and abs(cy-yloc)<=20:  
                    img = cv2.drawContours(img,[box],0,(0,255,0),2) 
                    
        cv2.imshow(basefile+'color', img)

        
        