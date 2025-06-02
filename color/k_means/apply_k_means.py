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

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_deep_features(img):
    img = cv2.resize(img, (224, 224))  
    img = preprocess_input(img)  
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features.flatten()


def load_images_from_folder(folder):
    """Carga imágenes desde la carpeta."""
    images = []
    filenames = []
    for filename in os.listdir(folder):
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

def extract_features_histogram_lab(img, bins=256):
    """Extrae características de color usando histogramas en el espacio LAB."""
    img = cv2.resize(img, (10, 10))  # Reducción de tamaño para eficiencia
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convertir BGR → LAB
    
    # Calcular histogramas por cada canal (L, A, B)
    hist_L = cv2.calcHist([img_lab], [0], None, [bins], [0, 256])
    hist_A = cv2.calcHist([img_lab], [1], None, [bins], [0, 256])
    hist_B = cv2.calcHist([img_lab], [2], None, [bins], [0, 256])
    
    # Normalizar histogramas para evitar influencia del tamaño de imagen
    hist_L /= hist_L.sum()
    hist_A /= hist_A.sum()
    hist_B /= hist_B.sum()
    
    # Concatenar los histogramas en un solo vector de características
    features = np.concatenate([hist_L.flatten(), hist_A.flatten(), hist_B.flatten()])

    return features

def extract_features_histogram_hs(img, bins=180):
    """
    Extrae características de color usando histogramas en el espacio HSV (H y S).
    """
    img = cv2.resize(img, (10, 10))  # Reducción de tamaño para eficiencia
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convertir BGR → HSV

    # Calcular histogramas para los canales H (Hue) y S (Saturation)
    hist_H = cv2.calcHist([img_hsv], [0], None, [bins], [0, 180])  # Hue va de 0 a 180
    hist_S = cv2.calcHist([img_hsv], [1], None, [bins], [0, 256])  # Saturación va de 0 a 255

    # Normalizar histogramas
    hist_H /= hist_H.sum()
    hist_S /= hist_S.sum()

    # Concatenar los histogramas en un solo vector de características
    features = np.concatenate([hist_H.flatten(), hist_S.flatten()])

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

def find_best_k(pixels, max_clusters=10):
    """Encuentra el número óptimo de clusters usando el coeficiente de silueta."""
    best_k = 2
    best_score = -1
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        score = silhouette_score(pixels, labels)
        print(f"K={k}, Silhouette Score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Best K {best_k}")
    return best_k

def save_images_to_clusters(filenames, labels):
    """Guarda las imágenes en carpetas según el cluster asignado."""
    output_folder = os.path.join("clusters")

    # Eliminar la carpeta de clusters si ya existía
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Crear subcarpetas para cada cluster
    for i, filename in enumerate(filenames):
        cluster_folder = os.path.join(output_folder, f"cluster_{labels[i]}")
        os.makedirs(cluster_folder, exist_ok=True)
        shutil.copy(os.path.join(folder_path, filename), os.path.join(cluster_folder, filename))

    print(f"Imágenes guardadas en {output_folder}")

def train_kmeans(pixels, k):
    """Entrena el modelo KMeans con el número óptimo de clusters."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    return kmeans

def plot_colors_lab(kmeans, k):
    cluster_centers = kmeans.cluster_centers_[:, -3:]
    cluster_folder = "cluster_colors"
    os.makedirs(cluster_folder, exist_ok=True)
    
    for i in range(k):
        lab_color = np.uint8([[cluster_centers[i]]])
        rgb_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2RGB)[0][0]
        rgb_color = [int(c) for c in rgb_color]
        
        img_color = np.zeros((100, 100, 3), dtype=np.uint8)
        img_color[:, :] = rgb_color
        save_path = os.path.join(cluster_folder, f"cluster_{i}.png")
        cv2.imwrite(save_path, img_color)
    print(f"Colores guardados en {cluster_folder}")


# Ruta de la carpeta con imágenes
folder_path = "../cropped_cars"

images, filenames = load_images_from_folder(folder_path)
image_features = np.array([extract_combined_features(img) for img in images])
best_k = find_best_k(image_features)
kmeans = train_kmeans(image_features, best_k)
labels = kmeans.predict(image_features)

save_images_to_clusters(filenames, labels)
plot_colors_lab(kmeans, best_k)
