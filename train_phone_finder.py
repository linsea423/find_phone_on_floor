# -*- coding: utf-8 -*-
"""
    This python code is used to generate sub-images for training SVM classifier,
        then train a SVM classifier to recognize phone,
        and finally save the SVM model to a file.

    Author: Linhai Li
"""

from phone_model import *
import cv2
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import log_loss
import pickle


if __name__ == "__main__":
    #get the folder where the original images are stored; consider the usage in shell and in IDE
    if len(sys.argv)<2:
        path = 'find_phone'
    else:
        path = sys.argv[1]
    path = os.path.abspath(path)

    #get the center of phone for each phone
    locs = pd.read_csv(os.path.join(path, 'labels.txt'), sep='\s+', header=None, names=['imgID', 'x', 'y'])

    #In the train phase, check if the training labels for phone classifier were already generated
    train_dir = os.path.join(os.path.curdir, 'train_img') #folder to store the training sub-imgs
    #If the training labels is created, delete the file in case duplicates added to the training label list
    trainLabel_fn = os.path.join(train_dir, 'trainLabels.csv') #files to store the labels for each sub-img

    #get the sub-images for training the classifier
    if not os.path.exists(train_dir):
        print('Creating sub-images for training SVM classifier...')
        for _, row in locs.iterrows(): #np.random.randint(0, 129, 2): #
            fn = os.path.join(path, row['imgID'])
            x = row['x']
            y = row['y']
            get_candidate_regions(fn, cx=x, cy=y, train_dir=train_dir, train=True)

    #Divide the datasets into training and validation. Theoretically this is not necessary as this was done in model_tuning
    #step, but just for fun to do this.
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

    trainD, valD, trainL, valL = train_test_split(trainHog, labels, test_size=0.2, random_state=888)

    #Now train a SVM model to classify phone and non-phone
    print('Training classifier to recognize phone...')
    mdl = SVC(C=0.0003, kernel='linear', gamma='scale', probability=True, random_state=1234)
    mdl.fit(trainD, trainL)

    #####Accuracy could be a metrics to evaluate the model as the classes are not so inbalanced; as a classification problem,
    #####log_loss is good to check as well. Log-loss may not be necessary close to each other as the dataset is small and
    #####the estimated probabilities in the training and validation datasets have similar distribution.
    #Print out the model accuracy for training dataset
    train_res = mdl.predict(trainD)
    train_prob =  mdl.predict_proba(trainD)
    print('Train accuracy: {:.2f}%'.format(sum(train_res==trainL)/len(trainL)*100))
    print('Train loss: {:.4f}'.format(log_loss(trainL, train_prob)))

    #The following three lines of code were only used for tuning the model. Not needed for final run
    val_res = mdl.predict(valD)
    val_prob = mdl.predict_proba(valD)
    print('Test accuracy: {:.2f}%'.format(sum(val_res==valL)/len(valL)*100))
    print('Test loss: {:.4f}'.format(log_loss(valL, val_prob)))

    #save the model
    print('Saving the model...')
    model_fn = os.path.join(os.path.abspath(os.path.curdir), 'trained_classifier.dat')
    pickle.dump(mdl, open(model_fn, 'wb'))

    print('Training process completed!')





