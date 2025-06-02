import os
import shutil
import re

# 📂 Rutas de origen y destino
dataset_original = "Dataset_completo_pregrado"
dataset_separado = "all_datasets"

# 📂 Directorios de salida
paths = {
    "cars": os.path.join(dataset_separado, "cars"),
    "plates": os.path.join(dataset_separado, "plates"),
    "plates_numbers_and_letters": os.path.join(dataset_separado, "plates_numbers_and_letters")
}

# 🏗️ Crear directorios si no existen
for key in paths:
    os.makedirs(os.path.join(paths[key], "images"), exist_ok=True)
    os.makedirs(os.path.join(paths[key], "labels"), exist_ok=True)

# 🏷️ Definir clases por índice
plate_class = {"0"}  # Placa
character_classes = {str(i) for i in range(1, 37)}  # Números y letras (1-36)
car_classes = {str(i) for i in range(37, 48)}  # Carros (37-47)

# 🔄 Recorrer los videos en Dataset_completo/
for video_folder in os.listdir(dataset_original):
    video_path = os.path.join(dataset_original, video_folder)

    if not os.path.isdir(video_path):
        continue

    img_folder = video_path  # Las imágenes están directamente en la carpeta del video
    label_folder = video_path  # Los labels también están en la misma carpeta

    # 🔍 Diccionario para encontrar imágenes por nombre sin extensión
    image_files = {re.sub(r"\.\w+$", "", f): f for f in os.listdir(img_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))}

    # 📂 Recorrer cada archivo de etiquetas
    for label_file in os.listdir(label_folder):
        if not label_file.endswith(".txt"):
            continue  # Ignorar archivos no TXT

        label_path = os.path.join(label_folder, label_file)

        # 📌 Obtener el nombre base para buscar la imagen
        base_name = re.sub(r"\.txt$", "", label_file)
        img_name = next((image_files[key] for key in image_files if base_name in key), None)

        if img_name is None:
            print(f"⚠️ No se encontró imagen para: {label_file}")
            continue  # Si no hay imagen, omitir

        img_path = os.path.join(img_folder, img_name)

        with open(label_path, "r") as file:
            lines = file.readlines()

        # 🚀 Separar etiquetas en categorías
        cars, plates, characters = [], [], []

        for line in lines:
            class_id, *coords = line.strip().split()
            class_id = class_id.strip()

            if class_id in car_classes:
                cars.append(f"{int(class_id) - 37} {' '.join(coords)}\n")
            elif class_id in plate_class:
                plates.append(line)
            elif class_id in character_classes:
                characters.append(f"{int(class_id) - 1} {' '.join(coords)}\n")

        # 📤 Guardar datos en la estructura correcta
        def save_data(category, data):
            if data:
                shutil.copy(img_path, os.path.join(paths[category], "images", os.path.basename(img_path)))
                new_label_path = os.path.join(paths[category], "labels", os.path.basename(label_path))
                with open(new_label_path, "w") as f:
                    f.writelines(data)

        save_data("cars", cars)
        save_data("plates", plates)
        save_data("plates_numbers_and_letters", characters)

print("✅ Separación completada")
