# -*- coding: utf-8 -*-
"""
    This python code is used to generate sub-images for training SVM classifier, 
        then train a SVM classifier to recognize phone, 
        and finally save the SVM model to a file.
        
    Author: Linhai Li
"""

from phone_model import *
import sys
import glob
import pandas as pd
import os
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

#plt.close('all')
#cv2.destroyAllWindows()

#get the folder where the original images are stored; consider the usage in shell and in IDE
if len(sys.argv)<2:
    path = 'find_phone'
else:
    path = sys.argv[1]
path = os.path.abspath(path)

#In the train phase, check if the training labels for phone classifier were already generated
train_dir = os.path.join(os.path.abspath(os.path.curdir), 'train_img') #folder to store the training sub-imgs

#If the training labels is created, delete the file in case duplicates added to the training label list
trainLabel_fn = os.path.join(train_dir, 'trainLabels.csv') #files to store the labels for each sub-img
if os.path.exists(trainLabel_fn):
    os.remove(trainLabel_fn)                    

####The following is used to generate images for final testing used in "find_phone.py" 
####and generate sub-imgs from the remaining images for training phone classifier 
loc_fn = glob.glob(os.path.join(path, '*.txt'))
locs = pd.read_csv(loc_fn[0], sep='\s+', header=None, names=['imgID', 'x', 'y'])

###This two lines of code were used for self testing model purpose
#test_set = locs.sample(n=10).reset_index(drop=True)
#locs = locs.drop(test_set.index).reset_index(drop=True)

for i in range(0,len(locs)): #np.random.randint(0, 129, 2): #
    fn = os.path.join(path, locs.loc[i,'imgID'])
    x = locs.loc[i,'x']
    y = locs.loc[i,'y']
    img_cl = phone_img(x, y)
    img_cl.get_candidate_regions(fn, train=True)
    # img_cl.img_proc(fn)
    
##Now train a SVM model to classify phone and non-phone
labelDF = pd.read_csv(trainLabel_fn)
cPimg = phone_img()
trainHog = []
labels = labelDF['label']
for fn in labelDF['filename']:
    img = cv2.imread(os.path.join(train_dir, fn))
    hist = cPimg.hog(img)
    trainHog.append(hist)
    
trainHog = np.float32(trainHog)
labels = np.array(labels, dtype=int)

#Devide the dataset into training, validation, and testing
random.seed(42)
shuf_idx = np.arange(0,len(labels))
random.shuffle(shuf_idx)
trainHog = trainHog[shuf_idx, :]
labels = labels[shuf_idx]

trainD, trainL = trainHog, labels

#The following line of code was used to training the model, it is not needed for final run
#trainD, testD, trainL, testL = train_test_split(trainHog, labels, test_size=0.2, random_state=888)

svm = SVC(C=2.67, kernel='linear', gamma=5.383, probability=True)    

svm.fit(trainD, trainL)

#Print out the model accuracy for training dataset
#train_res = svm.predict(trainD)
#print('Accuracy: ', sum(train_res==trainL)/len(trainL)*100, '%')

#save the model
model_fn = os.path.join(os.path.abspath(os.path.curdir), 'svm_classifier.dat')
pickle.dump(svm, open(model_fn, 'wb'))

#The following three lines of code were only used for tuning the model. Not needed for final run
#test_res = svm.predict(testD)
#test_prob = svm.predict_proba(testD)
#print(sum(test_res==testL)/len(testL))


