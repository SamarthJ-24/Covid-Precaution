from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import imutils
import joblib
import numpy as np
import time
import cv2
import os

prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath_F = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath_F)

clf = os.path.sep.join(["PCA", "Face_recognition.joblib"])
# clf = os.path.sep.join(["Face_recognition.joblib"])
pca = os.path.sep.join(["PCA", "pca.joblib"])
# pca = os.path.sep.join(["pca.joblib"])

clf = joblib.load(clf)
pca = joblib.load(pca)

vs = cv2.VideoCapture(0)
vs.set(3, 1280)
vs.set(4, 720)
time.sleep(2.0)

while True:
    ret, frame = vs.read()
    if frame is None:
        continue
    else:
        frame = imutils.resize(frame, width=500)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        face_locs = []
        label = []
        color = []
        ids = []
        counter = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                endX = startX + max(endX - startX, endY - startY)
                startX = startX - int((endY - startY) / 4)
                endX = endX - int((endY - startY) / 4)
                box = [startX, startY, endX, endY]
                face_locs.append(box)
                sub_face = frame[startY:endY, startX:endX]
                fname = "temp" + str(counter) + ".jpg"
                cv2.imwrite(os.path.join(str('Temp/'), fname), sub_face)
                counter += 1
        imagePaths = list(paths.list_images("Temp"))
        for imagePath in imagePaths:
            image = load_img(imagePath, color_mode="grayscale", target_size=(128, 128))
            image = img_to_array(image)
            image = preprocess_input(image)
            faces_data = np.array(image)
            faces = faces_data.reshape(1, faces_data.shape[0] * faces_data.shape[1])
            faces = pca.transform(faces)
            preds_face = clf.predict(faces)
            ids.append(preds_face)
            # os.remove(imagePath)
        for box, i in zip(face_locs, ids):
            x, y, z, w = box
            cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (z, w), thickness=1, color=(0, 255, 0))
    cv2.imshow("Test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()
