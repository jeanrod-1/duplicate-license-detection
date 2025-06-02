import os
import cv2
import numpy as np
from tqdm import tqdm

# === RUTAS ===
base_dir = "all_datasets"
cars_dir = os.path.join(base_dir, "cars")
plates_dir = os.path.join(base_dir, "plates")
output_dir = os.path.join(base_dir, "plates_from_cars")

splits = ["train", "val"]

# Crear carpetas de salida
for split in splits:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

def yolo_to_pixel(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return x1, y1, x2, y2

def pixel_to_yolo(x1, y1, x2, y2, crop_w, crop_h):
    x_c = ((x1 + x2) / 2) / crop_w
    y_c = ((y1 + y2) / 2) / crop_h
    w = (x2 - x1) / crop_w
    h = (y2 - y1) / crop_h
    return x_c, y_c, w, h

for split in splits:
    image_dir = os.path.join(plates_dir, "images", split)
    plate_label_dir = os.path.join(plates_dir, "labels", split)
    car_label_dir = os.path.join(cars_dir, "labels", split)

    image_filenames = os.listdir(image_dir)

    for filename in tqdm(image_filenames, desc=f"Procesando {split}"):
        if not filename.endswith(".jpg"):
            continue

        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer {img_path}")
            continue
        H, W, _ = img.shape

        base_name = filename.replace(".jpg", "")
        car_label_path = os.path.join(car_label_dir, f"{base_name}.txt")
        plate_label_path = os.path.join(plate_label_dir, f"{base_name}.txt")

        if not os.path.exists(car_label_path) or not os.path.exists(plate_label_path):
            continue

        # Leer boxes de carros
        with open(car_label_path, "r") as f:
            car_boxes = [list(map(float, line.strip().split()[1:])) for line in f.readlines()]

        # Leer boxes de placas
        with open(plate_label_path, "r") as f:
            plate_lines = f.readlines()

        plate_boxes = []
        for line in plate_lines:
            cls, *coords = line.strip().split()
            coords = list(map(float, coords))
            x1, y1, x2, y2 = yolo_to_pixel(coords, W, H)
            plate_boxes.append((int(cls), x1, y1, x2, y2))

        for i, car_box in enumerate(car_boxes):
            car_x1, car_y1, car_x2, car_y2 = yolo_to_pixel(car_box, W, H)
            car_crop = img[car_y1:car_y2, car_x1:car_x2]

            if car_crop.size == 0:
                continue

            crop_h, crop_w = car_crop.shape[:2]
            adjusted_plate_boxes = []

            for cls, px1, py1, px2, py2 in plate_boxes:
                # Verifica si la placa está dentro del carro
                if (px1 >= car_x1 and py1 >= car_y1 and
                    px2 <= car_x2 and py2 <= car_y2):
                    # Ajusta coordenadas al recorte
                    adj_x1 = px1 - car_x1
                    adj_y1 = py1 - car_y1
                    adj_x2 = px2 - car_x1
                    adj_y2 = py2 - car_y1
                    yolo_box = pixel_to_yolo(adj_x1, adj_y1, adj_x2, adj_y2, crop_w, crop_h)
                    adjusted_plate_boxes.append((cls, *yolo_box))

            # Si hay placas en este recorte, guardar
            if adjusted_plate_boxes:
                new_img_name = f"{base_name}_car{i}.jpg"
                new_txt_name = new_img_name.replace(".jpg", ".txt")
                out_img_path = os.path.join(output_dir, "images", split, new_img_name)
                out_txt_path = os.path.join(output_dir, "labels", split, new_txt_name)

                cv2.imwrite(out_img_path, car_crop)

                with open(out_txt_path, "w") as f:
                    for box in adjusted_plate_boxes:
                        f.write(f"{box[0]} {' '.join(f'{x:.10f}' for x in box[1:])}\n")
