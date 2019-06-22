# -*- coding: utf-8 -*-
"""
    Use the check the model performance or used to check a few things during the model development

    Author: Linhai Li
"""

from phone_model import *
import sys
import pandas as pd
import os
import numpy as np
import cv2
import pickle
import glob

from argparse import ArgumentParser

arg_parser =  ArgumentParser()
arg_parser.add_argument('check_type', nargs='+', default='model', help='Specify what to check')
args = vars(arg_parser.parse_args())

check_type = args['check_type']
if check_type[0] == 'model':
    check_type = 'model'

if __name__ == "__main__":

    #Load the train SVM classifier
    model_fn = os.path.join(os.path.abspath(os.path.curdir), 'trained_classifier.dat')
    mdl = pickle.load(open(model_fn, 'rb'))

    path = 'find_phone'

    #Read in the imgID and phone centers
    loc_fn = glob.glob(os.path.join(path, '*.txt'))
    test_set = pd.read_csv(loc_fn[0], sep='\s+', header=None, names=['imgID', 'x', 'y'])

    if check_type=='model':
        for i in range(len(test_set)):
            fn = os.path.join(path, test_set.loc[i,'imgID'])
            basefile = os.path.basename(fn)

            x = test_set.loc[i,'x']
            y = test_set.loc[i,'y']

            tempImg = cv2.imread(fn)
            w,h = tempImg.shape[1], tempImg.shape[0]

            candImgs, candRects = get_candidate_regions(fn, cx=x, cy=y, train=False)
            hog_hists = []
            for img in candImgs:
                hog_hist = hog(img)
                hog_hists.append(hog_hist)

            if len(hog_hists)>0:
                hog_hists = np.float32(hog_hists)
                probs = mdl.predict_proba(hog_hists)[:,1]
                idx = np.argmax(probs)

                prob = probs[idx]
                bbox = candRects[idx] #take the region with highest probability to be a phone
            else:
                bbox = None

            if len(candImgs)==0 or bbox is None:
                print(basefile + ' Not found!')
                cv2.imshow(basefile, tempImg)
            else:
                cx = bbox[0] + bbox[2]/2.0
                cy = bbox[1] + bbox[3]/2.0

                cv2.drawMarker(tempImg, (int(cx), int(cy)), (255,0,0), 1, 10, 2)

                #normalized to the shape
                cx = cx/w
                cy = cy/h

                distance = np.sqrt((cx-x)**2+(cy-y)**2)
                if distance>0.05:
                    cv2.imshow(basefile, tempImg)
                    print(basefile, ' too far by ', distance)
    elif isinstance(check_type, list):
        for i in check_type:
            fn = os.path.join(path, i)
            basefile = i

            x = test_set[test_set['imgID']==i]['x'].values[0]
            y = test_set[test_set['imgID']==i]['y'].values[0]

            explore_img(fn, x, y)
    else:
        sys.exit("Pass either 'model' or list of filenames to the images to check")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


