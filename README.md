# find_phone_on_floor
A very simple model was built to detect the center of a phone laying on the floor.

I built a model to detect the center of phone that laying on a floor. A summary and brief analysis is included in the repository. Feel free to read the summary and analysis to understand the model.

# Structure of the repository
find_phone: contains the original images and labels of the center of phones. The labels have the format of (img_id  x y). The x and y are the normalized coordinate of a phone center measured from top-left corner of the image

phone_model.py: the central image processing module

train_phone_finder.py: extracts sub-images from the big original images in "find_phone" folder and then trains a SVM classifier to classify a sub-image into phone vs. non-phone. This module automatically creates folder "train_img" to store the extracted sub-images. It also save the trained SVM classifier into file "svm_classifier.dat"

find_phone.py: predict the center of a phone given a image

# How to use this model
(assume you place "find_phone" folder, train_phone_finder.py, and find_phone.py at the same directory)

To train:

Linux & Windows: python train_phone_finder.py 

To predict:

Windows: python find_phone.py find_phone\100.jpg

Linux: python find_phone.py find_phone/100.jpg

# If questions, please read the summary and analysis PDF document or email me.





