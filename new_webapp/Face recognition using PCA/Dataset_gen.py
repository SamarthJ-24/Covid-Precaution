import cv2, os
# from sklearn.externals import joblib
import numpy as np
import csv

import glob

# run this script for all labels
dirList = glob.glob("dataset/Devarshi/*.jpg")
for img_path in dirList:
    file_name = img_path.split(os.path.sep)[-2]
    print(file_name)
    im = cv2.imread(img_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
    roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imshow("window", roi)

    data = []
    data.append("Devarshi")
    rows, cols = roi.shape

    # #Add pixel one-by-one into data Array.
    for i in range(rows):
        for j in range(cols):
            k = roi[i, j]
            if k > 100:
                k = 1
            else:
                k = 0

            data.append(k)

    # header =["label"]
    # for i in range(0,784):
    # 	header.append("pixel"+str(i))
    # print header

    with open('face_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    # scv2.imwrite("proc_images/0/"+file_name, roi)

    cv2.waitKey()