import os
import shutil
import re

# 📂 Rutas de origen y destino
dataset_original = "all_missing_images"
dataset_separado = "all_missing_datasets"

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

# 🔍 Diccionario para encontrar imágenes por nombre sin extensión
image_files = {re.sub(r"\.\w+$", "", f): f for f in os.listdir(dataset_original) if f.lower().endswith((".jpg", ".png", ".jpeg"))}

# 📂 Recorrer cada archivo de etiquetas en la carpeta raíz
for label_file in os.listdir(dataset_original):
    if not label_file.endswith(".txt"):
        continue  # Ignorar archivos no TXT

    label_path = os.path.join(dataset_original, label_file)

    # 📌 Obtener el nombre base para buscar la imagen
    base_name = re.sub(r"\.txt$", "", label_file)
    img_name = image_files.get(base_name)

    if img_name is None:
        print(f"⚠️ No se encontró imagen para: {label_file}")
        continue  # Si no hay imagen, omitir

    img_path = os.path.join(dataset_original, img_name)

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
            shutil.copy(img_path, os.path.join(paths[category], "images", img_name))
            new_label_path = os.path.join(paths[category], "labels", label_file)
            with open(new_label_path, "w") as f:
                f.writelines(data)

    save_data("cars", cars)
    save_data("plates", plates)
    save_data("plates_numbers_and_letters", characters)

print("✅ Separación completada")
