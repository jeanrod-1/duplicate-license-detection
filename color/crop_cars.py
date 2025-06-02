import os
import cv2
import torch
from ultralytics import YOLO

def process_images_and_labels(images_dir, output_folder, model_path):
    # Cargar modelo YOLO
    model = YOLO(model_path)

    # Crear directorios de salida
    os.makedirs(output_folder, exist_ok=True)

    # Obtener lista de imágenes
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)

        # Leer la imagen y sus labels
        image = cv2.imread(img_path)

        # Hacer la detección
        results = model(img_path)

        for j, result in enumerate(results):
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])  # Coordenadas
                cropped_car = image[y1:y2, x1:x2]  # Recortar 

                # Guardar imagen recortada
                new_img_name = f"{os.path.splitext(img_name)[0]}_car{i}.jpg"
                cropped_img_path = os.path.join(output_folder, new_img_name)
                cv2.imwrite(cropped_img_path, cropped_car)

    print("Procesamiento completado.")

# Parámetros
images_dir = "../all_datasets/cars/images/val"

model_path = "../yolo_training/runs/exp2-yolo11s/car_detection_all_datasets/weights/best.pt"

output_folder = "cropped_cars"

# Ejecutar función
process_images_and_labels(images_dir, output_folder, model_path)
