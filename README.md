# Covid-Precaution
### Python, Tkinter, OpenCV, Tensorflow, MySQL.

A computer vision system for monitoring face mask compliance and social distancing in real-time. Utilized state-of-the-art deep learning models including CAFFE for face detection, CNN for mask detection, YOLO-COCO for person detection and distance measurement, and Principal Component Analysis for facial recognition with face masks.

Face mask detection : For the face mask detection system we are loading an input image from disk and then detecting the face in the image and then using our face mask detector to classify the image as either with mask or without mask.

Social distance monitoring : Applying object detection to detect all people (and only people) in a video stream and then computing pairwise distances between all detected people and then based on these distances,  check the people who are not maintaining appropriate distance.

Face Recognition : Recognizing  masked faces requires making a system that can identify the available datasets of a masked face, i.e., facial features. The advanced detection technology includes various facial features left uncovered, such as eyes, eyebrows, and the bridge of the nose!
