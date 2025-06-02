import os
import shutil
import random

# Definir rutas (sin "new_dataset" porque ya estás dentro)
source_images = "extracted_frames"
source_labels = "data/labels/train"
dest_base = "cars"

# Crear estructura de carpetas esperada
for split in ["train", "val"]:
    os.makedirs(os.path.join(dest_base, "images", split), exist_ok=True)
    os.makedirs(os.path.join(dest_base, "labels", split), exist_ok=True)

# Obtener lista de imágenes y asociarlas con labels
images = [f for f in os.listdir(source_images) if f.endswith(".jpg")]
random.shuffle(images)  # Mezclar aleatoriamente para el split

split_idx = int(len(images) * 0.8)
train_images, val_images = images[:split_idx], images[split_idx:]

# Función para copiar y renombrar archivos
def copy_files(image_list, split):
    for image_file in image_list:
        # Extraer el identificador del frame
        frame_id = image_file.split("_")[-1].replace(".jpg", "")  # Extrae xxxxxx

        # Construir nuevo nombre para el label
        new_label_name = f"2025-02-26_puente_127_frame_{frame_id}.txt"

        # Rutas originales
        label_old_name = f"frame_{frame_id}.txt"
        label_old_path = os.path.join(source_labels, label_old_name)
        image_path = os.path.join(source_images, image_file)

        # Copiar imagen si existe
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(dest_base, "images", split, image_file))

        # Copiar y renombrar label si existe
        if os.path.exists(label_old_path):
            label_new_path = os.path.join(dest_base, "labels", split, new_label_name)
            shutil.copy(label_old_path, label_new_path)

# Copiar archivos a train y val con el nuevo nombre de labels
copy_files(train_images, "train")
copy_files(val_images, "val")

print("✅ Organización completada con renombrado de labels.")
