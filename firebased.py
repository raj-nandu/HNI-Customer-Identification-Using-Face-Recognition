# functions to connect with firebase
import pickle
import pyrebase
import cv2
import os

import shutil
from Main_file import *

# configuration
config = {
  "apiKey": "AIzaSyDtA3QrK4WXUGjCyzgXi_6vm7wJypXwy7U",
  "authDomain": "sbihackathon-cb717.firebaseapp.com",
  "databaseURL": "https://sbihackathon-cb717.firebaseio.com",
  "storageBucket": "sbihackathon-cb717.appspot.com",
  "serviceAccount": "./sbi_firestore.json"
}

firebase = pyrebase.initialize_app(config)


def add_to_firebase(identity, timestamp, face):
    # adds the identity, face and timestamp in firebase after detecting the HNI customer
    db = firebase.database()
    storage = firebase.storage()
    parent_id = db.child("customer").child(identity).child('parent').get().val()
    current_pending_requests = db.child("notifications").child(parent_id).shallow().get()

    if current_pending_requests.val() is None or identity not in current_pending_requests.val():
        cv2.imwrite('notification_images/' + identity + '.jpeg', face)
        data = {'timestamp': timestamp}
        db.child("notifications/" + parent_id).child(identity).set(data)
        storage.child("notifications/"+identity+".jpeg").put('notification_images/' + identity + '.jpeg')
        os.remove('notification_images/' + identity + '.jpeg')


def get_pickle_from_firebase():
    # downloads the database.pkl from firebase
    storage = firebase.storage()
    storage.child("database_backup/database.pkl").download('people/database.pkl')

def check_for_new():
    # check if a new user has been added in the database
    try:
        db = firebase.database()
        users = db.child("new_users").shallow().get()
        if users.val() is None:
            print("here")
            return False, None
        print("there")
        return bool(users), users
    except:
        return False, None


def upload_image(model, key):
    # converts image to encodings and stores it in database
    detector = FaceDetector()
    name = key
    images = [image for image in os.listdir('new_images/'+key)]
    vecs = []
    database = None
    for image in images:
        img = cv2.imread('new_images/' + key + '/' + image, 1)
        faces_cord = detector.detect(img)
        print("image aaya")
        if len(faces_cord):
            print("face aaya")
            faces = normalize_faces(img, faces_cord)
            vecs.append(model.get_vector(faces[0]))
    with open("people/database.pkl", 'rb') as data:
        database = pickle.load(data)
    database[1] += vecs
    for i in range(len(vecs)):
        database[0][len(database[0])] = name
        print("added name " + name + 'at' + str(len(database[1])))
    with open("people/database.pkl", 'wb') as data:
        pickle.dump(database, data)
        print("data dumped"+name)
    return database


def convert_to_vectors(users, model):
    # called whenever a new user has been detected by the thread that queries firebase every 30 seconds
    db = firebase.database()
    storage = firebase.storage()
    for key in users.val():
        os.mkdir('new_images/' + key)
        # makes a directory inside new_images with the name given by key
        for i in range(6):
            # downloads the images in the disl
            storage.child("customers/profile_pictures/"+key+"/"+str(i)+".jpeg").download('new_images/' + key + '/' + str(i)+".jpeg")
        # generates encodings of each downloaded image and stores the database
        database = upload_image(model, key)
        # deletes the downloaded images
        shutil.rmtree('new_images/' + key)
        # removes the user firebase
        db.child("new_users").child(key).remove()
    # uploads the updated database on firebase
    storage.child("database_backup/database.pkl").put('people/database.pkl')
    return database



