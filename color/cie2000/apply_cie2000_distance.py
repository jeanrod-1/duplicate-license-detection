from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color
import cv2
import numpy as np
import os
import pandas as pd

def get_average_color(image_path):
    """Calcula el color promedio de una imagen en espacio LAB."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo leer la imagen {image_path}")
            return None
       
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convertir BGR a LAB

        # segmentación, weighted, mediana, no senible a outliers, random sample concensus, otros algoritmos
        # Histograma en (L)AB?, 3d, bins
        # bajar resolución en espacio lab -- NN

        L_mean = np.mean(image_lab[:, :, 0])
        a_mean = np.mean(image_lab[:, :, 1])
        b_mean = np.mean(image_lab[:, :, 2])
        return LabColor(L_mean, a_mean, b_mean)
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

def rgb_to_lab(rgb_tuple):
    """Convierte un color RGB (0-255) a LAB."""
    # Normalizar RGB a rango 0-1
    r, g, b = rgb_tuple
    rgb_color = sRGBColor(r/255, g/255, b/255)
    # Convertir a LAB
    lab_color = convert_color(rgb_color, LabColor)
    return lab_color

def identify_closest_color(lab_color, reference_colors):
    """Identifica el color más cercano basado en la distancia CIEDE2000."""
    if lab_color is None:
        return "Error"
    closest_color = min(reference_colors, key=lambda name: delta_e_cie2000(lab_color, reference_colors[name]))
    return closest_color

def identify_colors_in_folder(folder_path, output_dir, output_file_name):
    """Identifica el color promedio de todas las imágenes en una carpeta y guarda los resultados en un CSV."""
    
    rgb_reference_colors = {
        "rojo": (255, 0, 0),          # Escoger colores de referencia en base a algoritmo no supervisado
        "azul": (0, 0, 255),           
        "blanco": (255, 255, 255),     
        "negro": (0, 0, 0),            
        "gris": (128, 128, 128),
        "gris claro": (192, 192, 192),
        "gris oscuro": (64, 64, 64),     
        "amarillo": (255, 255, 0),     
        "verde": (0, 128, 0),          
        "marrón": (139, 69, 19),
        "naranja": (255, 165, 0)
    }
    
    # Convertir los colores RGB a LAB
    reference_colors = {name: rgb_to_lab(rgb) for name, rgb in rgb_reference_colors.items()}
    
    print("Valores LAB de los colores de referencia:")
    for name, lab in reference_colors.items():
        print(f"{name}: L={lab.lab_l:.2f}, a={lab.lab_a:.2f}, b={lab.lab_b:.2f}")
   
    data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(folder_path, filename)
            avg_lab_color = get_average_color(image_path)
            identified_color = identify_closest_color(avg_lab_color, reference_colors)
            
            if avg_lab_color:
                min_distance = delta_e_cie2000(avg_lab_color, reference_colors[identified_color])
                data.append([
                    filename, 
                    identified_color, 
                    round(avg_lab_color.lab_l, 2),
                    round(avg_lab_color.lab_a, 2),
                    round(avg_lab_color.lab_b, 2),
                    round(min_distance, 2)
                ])
            else:
                data.append([filename, "Error", None, None, None, None])
   
    df = pd.DataFrame(data, columns=["imagen", "color_identificado", "L", "a", "b", "distancia"])
   
    os.makedirs(output_dir, exist_ok=True)  
   
    output_path = os.path.join(output_dir, output_file_name)
    df.to_csv(output_path, index=False)
    print(f"Resultados guardados en {output_path}")
    return df


folder_path = '../cropped_cars'
output_dir = "../results/cie2000"
output_file_name = 'color_output_mean.csv'
df_colors = identify_colors_in_folder(folder_path, output_dir, output_file_name)