# AI model
# We have used a pretrained FaceNet model since training a face recognition model needs too much computational resources

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import pickle
from threading import Thread
from firebased import *
import time
class FaceDetection(Thread):
    identities = []

    def run(self):
        # Thread to continuously check every 30s if any new user has been added to the database
        while True:
            if self.exit_flag:
                break
            yes, users = check_for_new()
            if yes:
                # if new user has been added then generate the encodings for all the uploaded images and store them locally in people/database.pkl file
                self.database = convert_to_vectors(users, self)
            time.sleep(30)

    def __init__(self):
        super().__init__()
        self.exit_flag = False
        self.FRmodel = faceRecoModel(input_shape=(3, 96, 96))
        self.FRmodel.compile(optimizer='adam', loss=self.triplet_loss, metrics=['accuracy'])
        # load the pretrained parameters
        load_weights_from_FaceNet(self.FRmodel)
        self.index_name = {}
        self.vectors = []
        # load the database of encoded images from people/database.pkl
        self.database = self.getData()
        

    def triplet_loss(self, y_true, y_pred, alpha=0.2):
        # loss function
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive))
        neg_dist = tf.reduce_sum(tf.square(anchor - negative))
        basic_loss = pos_dist - neg_dist + alpha
        loss = tf.maximum(basic_loss, 0)
        return loss

    def get_vector(self, image):
        # returns image encodings
        img = image[..., ::-1]
        img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
        x_train = np.array([img])
        embedding = self.FRmodel.predict_on_batch(x_train)
        return embedding

    def predict(self, image):
        # returns distance and identity
        encoding = self.get_vector(image)
        min_dist = 100
        for (no, db_enc) in enumerate(self.database[1]):
            dist = np.linalg.norm(db_enc - encoding)
            if dist < min_dist:
                min_dist = dist
                identity = self.database[0][no]

        if min_dist > 0.35:
            # threshold is set to 0.39 and can be decreased in order to reduce false positives
            print("Not in the database.")
            identity = 'unknown'
        else:
            print("it's " + str(identity) + ", the distance is " + str(min_dist))

        return min_dist, identity

    def getData(self):
        check_for_pickle = os.path.isfile("people/database.pkl")
        if check_for_pickle:
            # load database from disk if its available
            with open("people/database.pkl", 'rb') as data:
                self.database = pickle.load(data)
        else:
            # restores the database from firebase if database is not present in people folder
            get_pickle_from_firebase()
            with open("people/database.pkl", 'rb') as data:
                self.database = pickle.load(data)
        return self.database

    def dumpData(self):
        # dumps the database in a pickle file in people directory
        with open("people/database.pkl", 'wb') as data:
            pickle.dump(self.database, data)
            


