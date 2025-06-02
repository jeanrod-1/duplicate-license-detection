import os
import numpy as np
import cv2
import pickle
from collections import Counter

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def get_dominant_color_lab(image, k=3):
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_lab = cv2.resize(image_lab, (100, 100))
    pixels = image_lab.reshape(-1, 3)
    kmeans = cv2.kmeans(np.float32(pixels), k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_colors = kmeans[2].astype(int)
    labels = kmeans[1].flatten()
    color_counts = Counter(labels)
    most_common_color = dominant_colors[max(color_counts, key=color_counts.get)]
    return most_common_color

def extract_features_histogram(img, bins_L, bins_A, bins_B, bins_H, bins_S, bins_V):
    img_lab = cv2.cvtColor(cv2.resize(img, (10, 10)), cv2.COLOR_BGR2LAB)
    hist_lab = cv2.calcHist([img_lab], [0, 1, 2], None, [bins_L, bins_A, bins_B], [0, 256, 0, 256, 0, 256]).flatten()
    img_hsv = cv2.cvtColor(cv2.resize(img, (10, 10)), cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([img_hsv], [0, 1, 2], None, [bins_H, bins_S, bins_V], [0, 180, 0, 256, 0, 256]).flatten()
    return np.concatenate([hist_lab, hist_hsv])

def save_features_to_pickle(features_dict, output_folder, filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, filename), "wb") as f:
        pickle.dump(features_dict, f)
    print(f"✅ Features guardadas en: {filename}")

# Parámetros de los experimentos
experiments = [
    (8, 8, 8, 16, 16, 16), 
    (10, 8, 8, 16, 10, 10), 
    (12, 10, 10, 18, 12, 12),
    (14, 10, 10, 16, 10, 10), 
    (16, 12, 12, 18, 12, 12), 
    (10, 10, 10, 12, 8, 8),
    (12, 12, 12, 16, 8, 8), 
    (14, 14, 14, 18, 10, 10), 
    (8, 6, 6, 12, 6, 6),
    (16, 16, 16, 18, 12, 12)
]

image_folder = "../cropped_cars_v2"
output_folder = "features_output"
images, filenames = load_images_from_folder(image_folder)

for i, (bins_L, bins_A, bins_B, bins_H, bins_S, bins_V) in enumerate(experiments, 1):
    print(f"🔄 Ejecutando experimento {i}...")
    features_dict = {}
    for img, filename in zip(images, filenames):
        hist_features = extract_features_histogram(img, bins_L, bins_A, bins_B, bins_H, bins_S, bins_V)
        dominant_color = get_dominant_color_lab(img)
        features_dict[filename] = np.concatenate([hist_features, dominant_color])
    output_file = f"exp_{i}_bins_L{bins_L}_A{bins_A}_B{bins_B}_H{bins_H}_S{bins_S}_V{bins_V}.pkl"
    save_features_to_pickle(features_dict, output_folder, output_file)
print("✅ Todos los experimentos completados.")
