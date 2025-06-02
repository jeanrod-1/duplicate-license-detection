import os
import shutil
import random

# Directorios de origen
source_img_dir = "all_datasets/plates/images"
source_label_dir = "all_datasets/plates/labels"

# Directorios destino
train_img_dir = "all_datasets/plates/images/train"
val_img_dir = "all_datasets/plates/images/val"
train_label_dir = "all_datasets/plates/labels/train"
val_label_dir = "all_datasets/plates/labels/val"

# Crear las carpetas si no existen
for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(d, exist_ok=True)

# Obtener todas las imágenes (incluyendo .jpg, .jpeg y .png)
all_images = [f for f in os.listdir(source_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Mezclar aleatoriamente
random.seed(123)
random.shuffle(all_images)

# Definir proporción de train/val
split_ratio = 0.8
split_index = int(len(all_images) * split_ratio)

# Separar en train y val
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Función para mover archivos
def move_files(image_list, img_dest, label_dest):
    for img_file in image_list:
        img_path = os.path.join(source_img_dir, img_file)
        
        # Obtener el nombre del archivo sin la extensión
        base_name = os.path.splitext(img_file)[0]
        
        # Buscar su archivo de label (.txt)
        label_path = os.path.join(source_label_dir, base_name + ".txt")

        # Mover imagen
        shutil.move(img_path, os.path.join(img_dest, img_file))

        # Mover anotación si existe
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(label_dest, os.path.basename(label_path)))

# Ejecutar movimiento
move_files(train_images, train_img_dir, train_label_dir)
move_files(val_images, val_img_dir, val_label_dir)

print("✅ Archivos organizados en train y val correctamente.")
