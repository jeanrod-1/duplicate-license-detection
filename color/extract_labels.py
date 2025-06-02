import json
import os

# Cargar el JSON
json_path = "1400_labels.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Archivo de salida
output_txt = "labels.txt"

# Procesar los datos
with open(output_txt, "w", encoding="utf-8") as f:
    for entry in data:
        # Obtener el nombre original de la imagen desde la clave "image"
        image_path = entry.get("image", "").split("=")[-1]  # Extrae la parte final del path
        image_name = os.path.basename(image_path)  # Obtiene solo el nombre del archivo
        
        # Corregir el nombre de la imagen reemplazando "%20" con un espacio
        corrected_name = image_name.replace("%20", " ")
        
        # Verificar si la clave "choice" existe en el registro
        label = entry.get("choice")
        if label is None:
            print(f"Registro sin 'choice': {corrected_name}")
            continue
        
        # Normalizar la etiqueta y reemplazar "naranja" por "amarillo"
        label = label.lower()
        if label == "naranja":
            label = "amarillo"
        
        # Escribir en el archivo TXT en formato "nombre_imagen etiqueta"
        f.write(f"{corrected_name} {label}\n")

print(f"Archivo {output_txt} creado con éxito.")
