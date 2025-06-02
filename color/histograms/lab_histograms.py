import cv2
import numpy as np
import os
import pandas as pd

def get_dominant_color(image_path):
    """Extrae el color dominante en el espacio LAB de una imagen."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, None  # Si no se puede leer la imagen, retornar valores vacíos
    
    # Convertir la imagen a espacio LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Separar canales L, A y B
    L, A, B = cv2.split(lab_image)
    
    # Calcular histogramas de los canales A y B
    hist_L = cv2.calcHist([L], [0], None, [256], [0, 256])
    hist_A = cv2.calcHist([A], [0], None, [256], [0, 256])
    hist_B = cv2.calcHist([B], [0], None, [256], [0, 256])
    
    # Encontrar el valor más frecuente en cada canal
    dominant_L = int(np.argmax(hist_L))
    dominant_A = int(np.argmax(hist_A))
    dominant_B = int(np.argmax(hist_B))
    
    # Determinar el color aproximado
    color_identified = lab_to_color(dominant_A, dominant_B)

    return color_identified, dominant_L, dominant_A, dominant_B

def lab_to_color(A, B):
    """Asigna un nombre de color aproximado basado en valores A y B en LAB."""
    if A < 128 and B < 128:
        return "Verde"
    elif A > 128 and B < 128:
        return "Rojo"
    elif A < 128 and B > 128:
        return "Azul"
    elif A > 128 and B > 128:
        return "Amarillo"

def process_images(folder_path, output_dir, output_file_name):
    """Procesa todas las imágenes en un folder y guarda los resultados en un CSV."""
    data = []
    
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtrar imágenes
            image_path = os.path.join(folder_path, filename)
            color_identified, L, A, B = get_dominant_color(image_path)
            data.append([filename, color_identified, L, A, B])
    
    # Crear un DataFrame con los resultados
    df = pd.DataFrame(data, columns=['imagen', 'color_identificado', 'L', 'a', 'b'])
    
    # Guardar el DataFrame en un archivo CSV
    output_path = os.path.join(output_dir, output_file_name)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Resultados guardados en: {output_path}")

# Parámetros de entrada
folder_path = '../cropped_cars'
output_dir = "../results/histograms"
output_file_name = 'color_output.csv'

# Ejecutar el procesamiento
process_images(folder_path, output_dir, output_file_name)
