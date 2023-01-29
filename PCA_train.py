from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import joblib
from time import time
import pandas as pd
import numpy as np


# Displaying Original Images
def show_original_images(pixels):
    fig, axes = plt.subplots(6, 10, figsize=(9, 4),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        print(i)
        ax.imshow(np.array(pixels)[i].reshape(128, 128), cmap='gray')
    plt.show()


# Displaying Eigen faces
def show_eigenfaces(pca):
    fig, axes = plt.subplots(3, 7, figsize=(9, 4),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(128, 128), cmap='gray')
        ax.set_title("PC " + str(i + 1))
    plt.show()


# Reading dataset and creating dataframe
print("Reading CSV...")
df = pd.read_csv("face_data.csv")
targets = df["target"]
pixels = df.drop(["target"], axis=1)

show_original_images(pixels)

# Splitting Dataset into training and testing
print("Splitting training and testing")
x_train, x_test, y_train, y_test = train_test_split(pixels, targets)

# Performing PCA.
print("Performing PCA...")
pca = PCA(n_components=40).fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

show_eigenfaces(pca)

# Projecting Training data to PCA
print("Projecting the input data on the eigenfaces orthonormal basis")
Xtrain_pca = pca.transform(x_train)

##############

# Initializing Classifer and fitting training data
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid)
clf = clf.fit(Xtrain_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Performing testing and generating classification report
print("Predicting people's names on the test set")
t0 = time()
Xtest_pca = pca.transform(x_test)
y_pred = clf.predict(Xtest_pca)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred))
val = input("Save (Y/N) : ")
if val is "y":
    filename = "Face_recognition.joblib"
    joblib.dump(clf, filename)
    joblib.dump(pca, open("pca.joblib", "wb"))
    print("model saved.")
# # Testing for external images
# print("Reading test csvs...")
# test_d1_pixels = pd.read_csv("test_Devarshi_1.csv")
# test_d2_pixels = pd.read_csv("test_devarshi_2.csv")
# test_d3_pixels = pd.read_csv("test_Sakshi_1.csv")
# test_d4_pixels = pd.read_csv("test_Sakshi_2.csv")
# test_d5_pixels = pd.read_csv("test_bhut_1.csv")
# test_d6_pixels = pd.read_csv("test_Samarth_1.csv")
# test_d7_pixels = pd.read_csv("test_Unknown.csv")
# test_d8_pixels = pd.read_csv("test_mudra_1.csv")
#
# print("Performing pca on test...")
# test_1 = pca.transform(test_d1_pixels)
# test_2 = pca.transform(test_d2_pixels)
# test_3 = pca.transform(test_d3_pixels)
# test_4 = pca.transform(test_d4_pixels)
# test_5 = pca.transform(test_d5_pixels)
# test_6 = pca.transform(test_d6_pixels)
# test_7 = pca.transform(test_d7_pixels)
# test_8 = pca.transform(test_d8_pixels)
#
# print("Predicting...")
# test_pred_1 = clf.predict(test_1)
# test_pred_2 = clf.predict(test_2)
# test_pred_3 = clf.predict(test_3)
# test_pred_4 = clf.predict(test_4)
# test_pred_5 = clf.predict(test_5)
# test_pred_6 = clf.predict(test_6)
# test_pred_7 = clf.predict(test_7)
# test_pred_8 = clf.predict(test_8)
#
# print(test_pred_1)
# print(test_pred_2)
# print(test_pred_3)
# print(test_pred_4)
# print(test_pred_5)
# print(test_pred_6)
# print(test_pred_7)
# print(test_pred_8)
