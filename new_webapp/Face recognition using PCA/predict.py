from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import joblib
import cv2
import os
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=os.path.sep.join(["Testing", "test_0.jpg"]))
args = vars(ap.parse_args())

clf = os.path.sep.join(["PCA", "Face_recognition.joblib"])
pca = os.path.sep.join(["PCA", "pca.joblib"])

clf = joblib.load(clf)
pca = joblib.load(pca)

prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the input image from disk, clone it, and grab the image spatial
# dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()


for i in range(0, detections.shape[2]):
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    confidence = detections[0, 0, i, 2]

    # If confidence > 0.5, show box around face
    if confidence > 0.5:
        startX = startX - 50
        startY = startY - 50
        endX = endX + 50
        endY = endY + 50
        face = image[startY:endY, startX:endX]
        cv2.imwrite("temp.jpg", face)
        image = load_img("temp.jpg", grayscale=True, target_size=(512, 512))
        image = img_to_array(image)
        image = preprocess_input(image)
        faces_data = np.array(image)
        plt.imshow(faces_data, cmap='gray')
        plt.show()
        faces = faces_data.reshape(1, faces_data.shape[0] * faces_data.shape[1])
        faces = pca.transform(faces)
        preds = clf.predict(faces)
        print(preds)


