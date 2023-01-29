from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import pandas as pd
from imutils import paths
import matplotlib.pyplot as plt
import os

imagePaths = list(paths.list_images("dataset"))
data = []
labels = []

# For whole training dataset generation
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = load_img(imagePath, color_mode="grayscale", target_size=(128, 128))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

faces_image = np.array(data)
faces_target = np.array(labels)

plt.imshow(faces_image[20], cmap='gray')
plt.show()
print(faces_target[20])
n_row = 128
n_col = 128

faces_data = faces_image.reshape(faces_image.shape[0], faces_image.shape[1] * faces_image.shape[2])
n_samples = faces_image.shape[0]
X = faces_data
n_features = faces_data.shape[1]
# the label to predict is the id of the person
y = faces_target
n_classes = faces_target.shape[0]

df = pd.DataFrame(X)
df['target'] = pd.Series(y, index=df.index)

df.to_csv("face_data.csv", index=False)

