import os
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
from collections import Counter
from scipy.spatial import distance
import pickle

# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.models import Model

# base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
# model = Model(inputs=base_model.input, outputs=base_model.output)

# def extract_deep_features(img):
#     img = cv2.resize(img, (224, 224))  
#     img = preprocess_input(img)  
#     img = np.expand_dims(img, axis=0)
#     features = model.predict(img)
#     return features.flatten()


def load_images_from_folder(folder):
    """Carga imágenes desde la carpeta, solo archivos .jpg."""
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):  # Filtrar solo archivos .jpg
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def get_dominant_color_lab(image, k=3):
    """
    Obtiene el color dominante en la imagen en el espacio LAB utilizando K-Means.
    """
    # Convertir la imagen a LAB
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Redimensionar para reducir tiempo de cómputo
    image_lab = cv2.resize(image_lab, (100, 100))

    # Reestructurar la imagen para K-Means
    pixels = image_lab.reshape(-1, 3)

    # Aplicar K-Means
    kmeans = cv2.kmeans(
        np.float32(pixels),  # Datos de entrada (píxeles en LAB)
        k,  # Número de clusters (colores dominantes)
        None, 
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),  # Criterios de parada
        10,  # Número de inicializaciones de K-Means
        cv2.KMEANS_RANDOM_CENTERS  # Método de inicialización de centros
    )
    
    # Obtener los colores dominantes
    dominant_colors = kmeans[2].astype(int) # Centroides (k colores)
    labels = kmeans[1].flatten() # Etiquetas de cada píxel
    color_counts = Counter(labels) # Contar la cantidad de píxeles por cada color

    # Obtener el color más frecuente
    most_common_color = dominant_colors[max(color_counts, key=color_counts.get)]

    return most_common_color

def extract_features_histogram_lab(img, bins_L=10, bins_A=10, bins_B=10):
    """Extrae características de color usando histogramas en el espacio LAB."""
    img = cv2.resize(img, (10, 10))  # Reducción de tamaño para eficiencia
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convertir BGR → LAB
    
    # Calcular histogramas por cada canal (L, A, B)
    hist_3d = cv2.calcHist([img_lab], [0, 1, 2], None, [bins_L, bins_A, bins_B], [0, 256, 0, 256, 0, 256])
    
    # Concatenar los histogramas en un solo vector de características
    features = hist_3d.flatten()
    # features = hist_3d.reshape(1,-1)

    return features

def extract_features_histogram_hs(img, bins_H=16, bins_S=10, bins_V=10):
    """
    Extrae características de color usando histogramas en el espacio HSV (H y S).
    """
    img = cv2.resize(img, (10, 10))  # Reducción de tamaño para eficiencia
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convertir BGR → HSV

    # Calcular histogramas para los canales H (Hue) y S (Saturation)
    hist_3d = cv2.calcHist([img_hsv], [0, 1, 2], None, [bins_H, bins_S, bins_V], [0, 180, 0, 256, 0, 256]) 

    # Concatenar los histogramas en un solo vector de características
    features = hist_3d.flatten()
    # features = hist_3d.reshape(1,-1)

    return features

def extract_combined_features(image):
    """
    Combina el histograma y el color dominante en un solo vector.
    """
    hist_features_lab  = extract_features_histogram_lab(image) 
    dominant_color = get_dominant_color_lab(image) 
    # deep_features = extract_deep_features(image) 
    hist_features_hs = extract_features_histogram_hs(image)

    return np.concatenate([hist_features_lab, hist_features_hs, dominant_color])

def save_features_to_pickle(features_dict, output_folder="features_output", output_file="image_features_v2.pkl"):
    """Guarda las características en un archivo .pkl dentro de un folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, output_file)
    with open(output_path, "wb") as f:
        pickle.dump(features_dict, f)
    print(f"✅ Features guardadas en: {output_path}")

# 🔹 Cargar imágenes y extraer features
image_folder = "cropped_cars_v2"  
output_folder = "features_output" 

images, filenames = load_images_from_folder(image_folder)
features_dict = {}

print("🔄 Extrayendo características de las imágenes...")

for img, filename in zip(images, filenames):
    # print(f"📸 Procesando: {filename}")
    features_dict[filename] = extract_combined_features(img)

# 🔹 Guardar features en .pkl
save_features_to_pickle(features_dict, output_folder)

print("✅ Proceso completado")


# Reducir dimensionalidad:

# PCA
# SelectKBest
# Feature Importance


# Criterio de selección de tamaño de bins
# 



# Experimento de tamaño de bins en cojunto