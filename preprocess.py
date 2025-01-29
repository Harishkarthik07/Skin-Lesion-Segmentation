import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm

DATASET_PATH = "dataset/"
IMG_SIZE = 128  
def load_images():
    images = []
    filenames = []

    for file in tqdm(os.listdir(DATASET_PATH)):
        if file.endswith(".jpg") or file.endswith(".png"):
            image = plt.imread(os.path.join(DATASET_PATH, file))
            image = np.resize(image, (IMG_SIZE, IMG_SIZE, 3))  # Resize
            images.append(image / 255.0)  # Normalize
            filenames.append(file)
    
    return np.array(images), filenames

def apply_pca(images, n_components=50):
    flat_images = images.reshape(images.shape[0], -1)  # Flatten
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(flat_images)
    return reduced

def apply_kmeans(images, clusters=2):
    flat_images = images.reshape(images.shape[0], -1)
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    segmented = kmeans.fit_predict(flat_images)
    return segmented.reshape(images.shape[0], IMG_SIZE, IMG_SIZE)

if __name__ == "__main__":
    images, filenames = load_images()
    np.save("images.npy", images)
