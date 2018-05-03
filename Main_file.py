# Main file
# Recognize faces using a laptop's webcam and add them to the firebase database if they are HNI customers
import cv2
import cv2.face
from IPython.display import clear_output
from keras_model import *
from firebased import *
import time
import datetime


class FaceDetector(object):
    def __init__(self):
        self.classifier = cv2.CascadeClassifier('opencv_dependency/haarcascade_frontalface_default.xml')
    def detect(self, image, biggest_only=True):
        # Detects faces in the image passed and returns bounding boxes in the format (x, y, width, height)
        scale_factor = 1.2
        min_neighbours = 5
        min_size = (30, 30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
            cv2.CASCADE_SCALE_IMAGE
        faces_cord = self.classifier.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbours,
                                                      minSize=min_size, flags=flags)
        return faces_cord


class VideoCamera(object):
    def __init__(self, index=0):
        # starts the webcam
        # index specifies the device id(camera id), in our laptop it is 0
        self.video = cv2.VideoCapture(index)
        self.index = index
        print(self.video.isOpened())

    def __del__(self):
        # turns off the webcam
        self.video.release()

    def get_frame(self, in_grayscale=False):
        # captures images and return frame in grayscale if grayscale=True
        _, frame = self.video.read()
        original_frame = frame
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return original_frame, frame


def cut_faces(image, face_cord):
    # Crops the images according to the coordinates passed in face_cord
    faces = []
    for x, y, w, h in face_cord:
       # w_temp = int(0.2 * w/2)
        faces.append(image[y:y+h, x:x+w, :])
    return faces


def resize(images, size=(96, 96)):
    # resizes and returns an image of fixed size passed in the argument
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm

def draw_rectangles(image, cords):
    # draws bounding boxes around faces in the webcam's video stream
    for x, y, w, h in cords:
        w_temp = int(0.2 * w/2)
        cv2.rectangle(image, (x+w_temp, y),(x+w-w_temp, y+h), (150,150,0), 8)


def normalize_faces(frame, faces_cord, size=(96, 96)):
    # performs cropping and resizing of faces
    faces = cut_faces(frame, faces_cord)
    faces = resize(faces,size)
    return faces


def curr_time():
    # returns timestamp
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return timestamp

if __name__ == '__main__':
    # Create an object of the FaceDetection class
    model = FaceDetection()
    # start the thread to check if new users have been added
    model.start()
    webcam = VideoCamera(0)
    detector = FaceDetector()
    cv2.namedWindow("Demo", cv2.WINDOW_AUTOSIZE)

    while True:
        # get frames
        original_frame, frame = webcam.get_frame(True)
        # detect faces
        faces_cord = detector.detect(frame)
        if len(faces_cord):
            # This part is executed only if faces are detected in the frame
            # get normalized faces
            faces = normalize_faces(original_frame, faces_cord)
            for i, face in enumerate(faces):
                # perform prediction for each face
                min_dist, identity = model.predict(face)
                if identity != 'unknown':
                    # add the customer to firebase to send notification to the android user
                    t = time.time()
                    add_to_firebase(identity, curr_time(), face)
                    t2 = time.time()
                    print("Time taken to upload = " + str(t2-t))

                cv2.putText(original_frame, identity.capitalize(), (faces_cord[i][0], faces_cord[i][1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)

            clear_output(wait=True)
            # Draw bounding boxes around face
            draw_rectangles(original_frame, faces_cord)
            cv2.putText(original_frame, "Escape to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                        cv2.LINE_AA)
            cv2.imshow("Demo", original_frame)
            # wait for 20 milliseconds to capture next image and pressing escape button will terminate the main process
            if cv2.waitKey(20) & 0xFF == 27:
                model.exit_flag = True
                cv2.destroyAllWindows()
                break
    # Release webcam after termination of the main loop
    webcam.video.release()



