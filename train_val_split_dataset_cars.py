import os
import shutil
import random

# Directorio donde están las imágenes y anotaciones
source_dir = "dataset_cars"

# Directorios destino
train_img_dir = "dataset_cars/images/train"
val_img_dir = "dataset_cars/images/val"
train_label_dir = "dataset_cars/labels/train"
val_label_dir = "dataset_cars/labels/val"

# Crear las carpetas si no existen
for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(d, exist_ok=True)

# Obtener todas las imágenes
all_images = [f for f in os.listdir(source_dir) if f.endswith((".jpg", ".png"))]

# Mezclar aleatoriamente
random.seed(123)
random.shuffle(all_images)

# Definir proporción de train/val
split_ratio = 0.8
split_index = int(len(all_images) * split_ratio)

# Separar en train y val
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Mover archivos a sus respectivas carpetas
def move_files(image_list, img_dest, label_dest):
    for img_file in image_list:
        img_path = os.path.join(source_dir, img_file)
        label_path = os.path.join(source_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        # Mover imagen
        shutil.move(img_path, os.path.join(img_dest, img_file))
        
        # Mover anotación si existe
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(label_dest, os.path.basename(label_path)))

#Ejecutar movimiento
move_files(train_images, train_img_dir, train_label_dir)
move_files(val_images, val_img_dir, val_label_dir)

print("Archivos organizados en train y val correctamente.")
