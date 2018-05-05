# HNI-Customer-Identification-Using-Face-Recognition
## Description
Every bank has a set of HNI(High net worth individual) customers. It's difficult for the bank staff to distinguish these customers from general customers. This project uses face recognition technology to identify HNI customers present in the bank. Whenever an HNI customer is detected, a picture of the customer taken via CCTV(Laptop's webcam is used in the code as of now) camera and also a picture from the database is sent to the bank staff via a notification generated on an android application. The bank staff then compares both these images and has two choices 
* Attend the customer. 
* Reject the notification(Useful if at all any non HNI customer is recognized falsely as an HNI customer).


## The face recognition pipeline is explained as follows
* Faces are detected using openCV.
* All faces detected in a frame are then cropped and storred in a list.
* Each face is passed through a siamese network which generates a 128 dimensional encodings of each detected face. These encodings are compared with encodings in the database and distance between two encodings is calculated. If the distance is <= 0.35, then it's a match and the image is added to the firebase database.

Notification is sent to the android app using Firebase.

## Dependencies
* Tensorflow 1.4.0
* Keras 2.1.2
* OpenCV
* pyrebase

## Description of Each Project File
* firebased.py - Contains all functions that are used to query Firebase.
* fr_utils.py - Contains helper funtions. PART OF THIS CODE IS USING CODE FROM VICTOR SY WANG: https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py  and Coursera's deeplearning.ai course
* inception_blocks_v2.py - Architechture of the inception model is defined here
* keras_model.py - This file contains all the functions such as get_vector(), predict(), getData(), dumpData() and a thread to generate encodings of newly entered images of HNI customers and store them in the database.
* Main_file.py - This file needs to be run to start the face recognition system.


## Siamese Network Architechture
* A pretrained inception model with 128 output units has been used to recognize faces(Since training a face recognition model from scratch needs too much computational power).
* Loss function - Triplet Loss 

## Database
We used firebase as our project's databse.


## Collaborators
* Gagandeep Uppal
* Rushabh Doshi
