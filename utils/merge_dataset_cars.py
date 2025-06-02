import os
import shutil
import re

# 📂 Rutas de los datasets
dataset_cars = "dataset_cars_pregrado"
dataset_target = "all_datasets/cars"
resized_file = os.path.join(dataset_cars, "resized.txt")

# 📂 Directorios destino
images_target = os.path.join(dataset_target, "images")
labels_target = os.path.join(dataset_target, "labels")

# 🏗️ Crear directorios si no existen
os.makedirs(images_target, exist_ok=True)
os.makedirs(labels_target, exist_ok=True)

# 🔄 Leer archivo resized.txt y mapear nombres originales
name_mapping = {}
if os.path.exists(resized_file):
    with open(resized_file, "r") as file:
        for line in file:
            parts = re.split(r" -> ", line.strip())
            if len(parts) == 2:
                original_name = re.sub(r" \[.*\]", "", parts[0].split("/")[-1])
                new_name = re.sub(r" \[.*\]", "", parts[1].split("/")[-1])
                name_mapping[new_name] = original_name

# 📂 Procesar train y val en dataset_cars
for split in ["train", "val"]:
    images_folder = os.path.join(dataset_cars, "images", split)
    labels_folder = os.path.join(dataset_cars, "labels", split)
    
    if not os.path.exists(images_folder):
        continue

    for img_file in os.listdir(images_folder):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # Ignorar archivos que no sean imágenes
        
        original_name = name_mapping.get(img_file, None)
        if original_name is None:
            continue
        
        # Verificar si ya existe en all_datasets/cars/images
        if os.path.exists(os.path.join(images_target, original_name)):
            continue  # Evitar duplicados

        # 📌 Copiar imagen
        shutil.copy(os.path.join(images_folder, img_file), os.path.join(images_target, original_name))

        # 📌 Buscar y copiar etiqueta correspondiente
        label_file = os.path.splitext(img_file)[0] + ".txt"  # Reemplaza la extensión por .txt
        label_source_path = os.path.join(labels_folder, label_file)
        label_target_path = os.path.join(labels_target, os.path.splitext(original_name)[0] + ".txt")

        if os.path.exists(label_source_path):
            shutil.copy(label_source_path, label_target_path)

print("✅ Integración completada. Revisa 'all_datasets/cars'.")
