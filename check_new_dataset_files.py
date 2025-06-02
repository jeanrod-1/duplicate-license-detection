import os

def count_yolo_objects(label_dir):
    count = 0
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            with open(os.path.join(label_dir, file), 'r') as f:
                lines = f.readlines()
                count += len(lines)
    return count

def count_files_with_labels(images_dir, labels_dir):
    with_labels = 0
    without_labels = 0
    for file in os.listdir(images_dir):
        if file.endswith('.jpg'):
            label_file = os.path.join(labels_dir, file.replace('.jpg', '.txt'))
            if os.path.exists(label_file):
                with_labels += 1
            else:
                without_labels += 1
    return with_labels, without_labels

# Rutas
base = "all_datasets"
cars_train_labels = os.path.join(base, "cars/labels/train")
cars_val_labels = os.path.join(base, "cars/labels/val")

plates_train_labels = os.path.join(base, "plates/labels/train")
plates_val_labels = os.path.join(base, "plates/labels/val")

plates_from_cars_train_images = os.path.join(base, "plates_from_cars/images/train")
plates_from_cars_train_labels = os.path.join(base, "plates_from_cars/labels/train")

plates_from_cars_val_images = os.path.join(base, "plates_from_cars/images/val")
plates_from_cars_val_labels = os.path.join(base, "plates_from_cars/labels/val")

# Conteo original
total_cars = count_yolo_objects(cars_train_labels) + count_yolo_objects(cars_val_labels)
total_plate_labels = count_yolo_objects(plates_train_labels) + count_yolo_objects(plates_val_labels)

# Conteo nuevo dataset
train_with_labels, train_without_labels = count_files_with_labels(plates_from_cars_train_images, plates_from_cars_train_labels)
val_with_labels, val_without_labels = count_files_with_labels(plates_from_cars_val_images, plates_from_cars_val_labels)

# Resultados
print("Resumen del dataset:\n")
print(f"🚗 Total de carros detectados (con label) en 'cars': {total_cars}")
print(f"🔢 Total de placas detectadas (con label) en 'plates': {total_plate_labels}")
print(f"📦 Total de imágenes generadas en 'plates_from_cars/train': {train_with_labels + train_without_labels}")
print(f"📦 Total de imágenes generadas en 'plates_from_cars/val':   {val_with_labels + val_without_labels}")
print(f"📄 Total labels generados en 'plates_from_cars': {train_with_labels + val_with_labels}")
print(f"⚠️  Imágenes sin label en 'plates_from_cars/train': {train_without_labels}")
print(f"⚠️  Imágenes sin label en 'plates_from_cars/val':   {val_without_labels}")
