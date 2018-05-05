# HNI-Customer-Identification-Using-Face-Recognition
Every bank has a set of HNI(High net worth individual) customers. It's difficult for the bank staff to distinguish these customers from general customers. This project uses face recognition technology to identify HNI customers present in the bank. Whenever an HNI customer is detected, a picture of the customer taken via CCTV(Laptop's webcam is used in the code as of now) camera and also a picture from the database is sent to the bank staff via a notification generated on an android application. The bank staff then compares both these images and has two choices 1) Attend the customer. 2) Reject the notification(Useful if at all any non HNI customer is recognized falsely as an HNI customer).


The face recognition pipeline is explained as follows
Step 1: Faces are detected using openCV.
Step 2: All faces detected in a frame are then cropped and storred in a list.
Step 3: Each face is passed through a siamese network which generates a 128 dimensional encodings of each detected face. These encodings are compared with encodings in the database and distance between two encodings is calculated. If the distance is <= 0.35, then it's a match and the image is added to the firebase database.

Notification is sent to the android app using Firebase.
