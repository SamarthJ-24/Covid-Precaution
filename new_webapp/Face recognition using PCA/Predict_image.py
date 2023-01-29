from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import joblib
import numpy as np
import argparse
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=os.path.sep.join(["Testing", "test_3.jpg"]))
args = vars(ap.parse_args())

clf = os.path.sep.join(["PCA", "Face_recognition.joblib"])
pca = os.path.sep.join(["PCA", "pca.joblib"])

clf = joblib.load(clf)
pca = joblib.load(pca)


def facecrop(image):
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    img = cv2.imread(image)

    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe, 1.1, 7)
    counter = 0
    face_locs = []
    ids = []
    for f in faces:
        x, y, w, h = [v for v in f]
        x = x - 100
        y = y - 100
        w = w + 200
        h = h + 200
        box = [x, y, x+w, y+h]
        face_locs.append(box)
        sub_face = img[y:y+h, x:x+w]
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
        preds = clf.predict(faces)
        ids.append(preds)
    for box, i in zip(face_locs, ids):
        x, y, z, w = box
        cv2.rectangle(img, (x, y), (z, w), thickness=10, color=(0, 255, 0))
        cv2.putText(img, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
        img = cv2.resize(img, (int(img.shape[1]*50/100), int(img.shape[0]*50/100)))

    cv2.moveWindow("Output", 40, 30)
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


facecrop(args["image"])


