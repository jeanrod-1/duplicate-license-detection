import cv2
import numpy as np
import os
import pandas as pd
from collections import Counter
from scipy.spatial import distance
from sklearn.metrics import silhouette_score

# Diccionario de colores en espacio RGB
color_labels = {
    "rojo": (255, 0, 0),
    "azul": (0, 0, 255),
    "blanco": (255, 255, 255),
    "negro": (0, 0, 0),
    "gris": (128, 128, 128),
    "gris claro": (192, 192, 192),
    "gris oscuro": (64, 64, 64),
    "amarillo": (255, 255, 0),
    "verde": (0, 128, 0),
    "marrón": (139, 69, 19),
    "naranja": (255, 165, 0),
}

# Convertir colores de referencia a LAB
color_labels_lab = {}
for name, rgb in color_labels.items():
    rgb_color = np.uint8([[list(rgb)]])
    lab_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2LAB)[0][0]
    color_labels_lab[name] = tuple(lab_color)

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
    dominant_colors = kmeans[2].astype(int)

    # Contar la cantidad de píxeles por cada color
    labels = kmeans[1].flatten()
    color_counts = Counter(labels) 

    # Obtener el color más frecuente
    most_common_color = dominant_colors[max(color_counts, key=color_counts.get)]

    return most_common_color

def closest_color(lab_color):
    """
    Encuentra el color más cercano en la lista predefinida usando distancia euclidiana en LAB.
    """
    color_diffs = {
        name: distance.euclidean(lab_color, lab_ref) 
        for name, lab_ref in color_labels_lab.items()
    }
    
    closest_name = min(color_diffs, key=color_diffs.get)
    closest_dist = color_diffs[closest_name]
    
    return closest_name, closest_dist

def process_images_in_folder(folder_path, output_dir, output_file_name):
    """
    Procesa todas las imágenes en una carpeta y guarda los resultados en un CSV.
    """
    results = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtrar solo imágenes
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"⚠️ No se pudo leer la imagen: {file}")
                continue

            # Obtener el color dominante en LAB
            dominant_lab = get_dominant_color_lab(image)
            L, a, b = dominant_lab

            # Buscar el color más cercano
            detected_color, dist_to_closest = closest_color(dominant_lab)

            # Guardar resultado
            results.append({
                "imagen": file, 
                "color_detectado": detected_color, 
                "L": L, "a": a, "b": b, 
                "distancia": dist_to_closest
            })
            # print(f"✅ Procesado {file}: {detected_color}")

    # Guardar resultados en un archivo CSV
    df = pd.DataFrame(results)
    
    os.makedirs(output_dir, exist_ok=True)  
   
    output_path = os.path.join(output_dir, output_file_name)
    df.to_csv(output_path, index=False)
    print(f"\n📄 Resultados guardados en: {output_path}")
    return df 


# Parámetros de entrada
carpeta_imagenes = '../cropped_cars'
output_dir = "../results/k_means_per_image"
output_file_name = 'color_output.csv'

# Procesar imágenes y guardar resultados
df_resultado = process_images_in_folder(carpeta_imagenes, output_dir, output_file_name)
