import os
import numpy as np
import cv2
import pickle
from collections import Counter

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