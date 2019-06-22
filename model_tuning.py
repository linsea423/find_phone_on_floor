# -*- coding: utf-8 -*-
"""
    This is used to tune the model

    Author: Linhai Li
"""

from phone_model import *
import cv2
import sys
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import log_loss, make_scorer
from sklearn.ensemble import RandomForestClassifier
import shutil

#Define hyper-parameters; These can be tuned
#NOTE: I already run the model tuning to find good values. To save time on running this (in case you would run),
#I only put the good values I found in the list
MAXBOXES = [5]
LTHRES = [0.1]
HTHRES = [0.95]


if __name__ == "__main__":
    #get the folder where the original images are stored; consider the usage in shell and in IDE
    if len(sys.argv)<2:
        path = 'find_phone'
    else:
        path = sys.argv[1]
    path = os.path.abspath(path)

    #get the center of phone for each phone
    locs = pd.read_csv(os.path.join(path, 'labels.txt'), sep='\s+', header=None, names=['imgID', 'x', 'y'])

    #Hold out datasets to test the final performance on getting the center of phone
    test_set = locs.sample(n=8, random_state=123).reset_index(drop=True)
    locs = locs.drop(test_set.index).reset_index(drop=True)

    #In the train phase, check if the training labels for phone classifier were already generated
    train_dir = os.path.join(os.path.curdir, 'tuning_img') #folder to store the training sub-imgs
    #If the training labels is created, delete the file in case duplicates added to the training label list
    trainLabel_fn = os.path.join(train_dir, 'trainLabels.csv') #files to store the labels for each sub-img

    for MB in MAXBOXES:
        for LT in LTHRES:
            for HT in HTHRES:
                print('Tuning... MB: {:d}, LT: {:.4f}, HT: {:.4f}'.format(MB, LT, HT))

                if os.path.exists(train_dir):
                    shutil.rmtree(train_dir, ignore_errors=True)

                #get the sub-images for training the classifier
                for _, row in locs.iterrows():
                    fn = os.path.join(path, row['imgID'])
                    x = row['x']
                    y = row['y']
                    get_candidate_regions(fn, cx=x, cy=y, maxBoxes=MB, LThres=LT, HThres=HT, train_dir=train_dir, train=True)

                #Divide the datasets into training and validation.
                labelDF = pd.read_csv(trainLabel_fn)
                trainHog = []
                labels = labelDF['label']
                for fn in labelDF['filename']:
                    img = cv2.imread(os.path.join(train_dir, fn))
                    hog_hist = hog(img)
                    trainHog.append(hog_hist)

                trainHog = np.float32(trainHog)
                labels = np.array(labels, dtype=int)

                random.seed(42)
                shuf_idx = np.arange(0,len(labels))
                random.shuffle(shuf_idx)
                trainHog = trainHog[shuf_idx, :]
                labels = labels[shuf_idx]

                print('   Fraction of phone images in all data: {:.4f}'.format(sum(labels==1)/len(labels)))
                trainD, valD, trainL, valL = train_test_split(trainHog, labels, test_size=0.2, random_state=888)
                print('   Fraction of phone images in training data: {:.4f}'.format(sum(trainL==1)/len(trainL)))
                print('   Fraction of phone images in validation data: {:.4f}'.format(sum(valL==1)/len(valL)))

                ####Try different classifiers, but it turns out SVM performs better and much faster in general
                parameters = {'C': [0.0003, 0.001, 0.003, 0.01], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2,3,4]}
                est = SVC(gamma='scale', probability=True, random_state=1234)

                #parameters = {'n_estimators': [30, 50, 100, 150, 200], 'max_depth': [4, 8, 10, 12, 16, 20]}
                #est = RandomForestClassifier(n_jobs=-1, bootstrap=False)

                loss_scorer = make_scorer(log_loss, greater_is_better=False)
                clf = GridSearchCV(est, parameters, cv=5, n_jobs=-1, scoring=loss_scorer)
                clf.fit(trainD, trainL)
                mdl = clf.best_estimator_
                print('   Best parameters: ', end='')
                print(clf.best_params_)

                #####Accuracy could be a metrics to evaluate the model as the classes are not so inbalanced; as a classification problem,
                #####log_loss is good to check as well. Log-loss may not be necessary close to each other as the dataset is small and
                #####the estimated probabilities in the training and validation datasets have similar distribution.
                #Print out the model accuracy for training dataset
                train_res = mdl.predict(trainD)
                train_prob =  mdl.predict_proba(trainD)
                print('   Train accuracy: {:.2f}%'.format(sum(train_res==trainL)/len(trainL)*100))
                print('   Train loss: {:.4f}'.format(log_loss(trainL, train_prob)))

                #The following three lines of code were only used for tuning the model. Not needed for final run
                val_res = mdl.predict(valD)
                val_prob = mdl.predict_proba(valD)
                print('   Test accuracy: {:.2f}%'.format(sum(val_res==valL)/len(valL)*100))
                print('   Test loss: {:.4f}'.format(log_loss(valL, val_prob)))

                #See how the entire model performs to get the center of the phone on the hold-out dataset
                i_found = 0
                for _, row in test_set.iterrows():
                    imgID = row['imgID']
                    fn = os.path.join(path, imgID)
                    x = row['x']
                    y = row['y']
                    candImgs, candRects = get_candidate_regions(fn, maxBoxes=MB, LThres=LT, HThres=HT, train=False)

                    tempImg = cv2.imread(fn)
                    w,h = tempImg.shape[1], tempImg.shape[0]

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

                    if len(candImgs)>0 and bbox is not None:
                        cx = bbox[0] + bbox[2]/2.0
                        cy = bbox[1] + bbox[3]/2.0

                        #normalized to the shape
                        cx = cx/w
                        cy = cy/h

                        if np.sqrt((cx-x)**2+(cy-y)**2)<=0.05:
                            i_found += 1
                print('   {} out of 8 phones found in the hold-out dataset'.format(i_found))



