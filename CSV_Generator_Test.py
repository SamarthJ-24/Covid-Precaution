from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = []
# Loading image from disk
image = load_img("test_devarshi_2.jpg", grayscale=True, target_size=(512, 512))
image = img_to_array(image)
image = preprocess_input(image)

data.append(image)
faces_image = np.array(data)

# Plotting processed image
plt.imshow(faces_image[0], cmap='gray')
plt.show()
n_row = 512
n_col = 512

faces_data = faces_image.reshape(faces_image.shape[0], faces_image.shape[1] * faces_image.shape[2])
n_samples = faces_image.shape[0]
X = faces_data
n_features = faces_data.shape[1]

df = pd.DataFrame(X)

df.to_csv("test_devarshi_2.csv", index=False)
