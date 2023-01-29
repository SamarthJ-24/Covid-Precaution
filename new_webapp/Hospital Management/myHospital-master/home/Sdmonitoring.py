# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import time


NMS_THRESH = 0.3
MIN_CONF = 0.3
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNet(weightsPath, configPath)

# check if we are going to use GPU

print("[INFO] setting preferable backend and target to CUDA...")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


class SocialDistancing(object):
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.vs.set(3, 1280)
        self.vs.set(4, 720)
        time.sleep(2.0)

    def __del__(self):
        cv2.destroyAllWindows()

    def detect_people(self, frame, net, ln, personIdx=0):
        # grab the dimensions of the frame and  initialize the list of
        # results
        (H, W) = frame.shape[:2]
        results = []

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, centroids, and
        # confidences, respectively
        boxes = []
        centroids = []
        confidences = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter detections by (1) ensuring that the object
                # detected was a person and (2) that the minimum
                # confidence is met
                if classID == personIdx and confidence > MIN_CONF:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # centroids, and confidences
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # update our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

        # return the list of results
        return results

    def startsd(self):

        ret, frame = self.vs.read()
        frame = imutils.resize(frame, width=700)
        ret, jpeg = cv2.imencode('.jpg', frame)

        results = self.detect_people(frame, net, ln)

        # initialize the set of indexes that violate the minimum social
            # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if D[i, j] < 50:
                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        text = "Social Distancing Violation count: {}".format(len(violate))
        cv2.putText(frame, text, (0, frame.shape[0] - 500), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes(), len(violate)

            # if an output video file path has been supplied and the video
            # writer has not been initialized, do so now
            # if args["output"] != "" and writer is None:
            #     # initialize our video writer
            #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            #     writer = cv2.VideoWriter(args["output"], fourcc, 25,
            #                              (frame.shape[1], frame.shape[0]), True)

            # if the video writer is not None, write the frame to the output
            # video file
            # if writer is not None:
            #     writer.write(frame)
