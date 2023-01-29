from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import imutils
import cv2
import os
import time
import numpy as np
import joblib
from django.conf import settings

# load our serialized face detector model from disk
prototxtPath = os.path.sep.join([settings.BASE_DIR, "face_detector/deploy.prototxt"])
weightsPath = os.path.sep.join([settings.BASE_DIR, "face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model(os.path.join(settings.BASE_DIR, 'face_detector/mask_detector.model'))

clf = os.path.sep.join([settings.BASE_DIR, "PCA/Face_recognition.joblib"])
pca = os.path.sep.join([settings.BASE_DIR, "PCA/pca.joblib"])

clf = joblib.load(clf)
pca = joblib.load(pca)


class MaskDetect(object):
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.vs.set(3, 1280)
        self.vs.set(4, 720)
        time.sleep(2.0)

    def __del__(self):
        cv2.destroyAllWindows()

    def detect_and_predict_mask(self, frame, faceNet, maskNet):

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                endX = startX + max(endX - startX, endY - startY)
                startX = startX - int((endY - startY) / 6)
                endX = endX - int((endY - startY) / 6)

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return locs, preds

    def get_frame(self):
        ret, frame = self.vs.read()
        frame = imutils.resize(frame, width=400)
        ret, jpeg = cv2.imencode('.jpg', frame)
        (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)
        counter = 0
        for (box, pred) in zip(locs, preds):

            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label_mask = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            if label == str("Mask"):

                sub_face = frame[startY:endY, startX:endX]
                fname = "temp" + str(counter) + ".jpg"
                cv2.imwrite(os.path.join(str('Temp/'), fname), sub_face)
                image = load_img(os.path.join(str('Temp/'), fname), color_mode="grayscale", target_size=(128, 128))
                image = img_to_array(image)
                image = preprocess_input(image)
                faces_data = np.array(image)
                faces = faces_data.reshape(1, faces_data.shape[0] * faces_data.shape[1])
                faces = pca.transform(faces)
                preds_face = clf.predict(faces)
                counter += 1
            else:
                preds_face = "Unknown"

            cv2.putText(frame, label_mask, (startX, startY - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.putText(frame, "ID : " + str(preds_face), (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()
