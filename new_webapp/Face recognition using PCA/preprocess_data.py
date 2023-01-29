import time
import cv2
import os
import imutils
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=False,
                help="path to where the face cascade resides", default="haarcascade_frontalface_default.xml")
ap.add_argument("-o", "--output", required=False,
                help="path to output directory", default="dataset/")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])

imagePaths = list(paths.list_images("dataset"))

for img_path in imagePaths:
    label = img_path.split(os.path.sep)[-2]

    im = cv2.imread(img_path)
    rects = detector.detectMultiScale(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))
    # loop over the face detections and draw them on the frame
    for (x, y, w, h) in rects:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("window", im)

    cv2.waitKey()