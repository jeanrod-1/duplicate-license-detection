import cv2
import torch
import pytesseract
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import pickle
import joblib
import pandas as pd
import itertools
import re
import string

from color_features import extract_features_histogram, get_dominant_color_lab
from yolo_utils import iou, remove_duplicate_boxes
from ocr_utils import preprocess_plate_for_ocr, clean_plate_text
from summarizer import summarize_similar_plates

# ================================
# Paths a tus modelos entrenados
# ================================
CAR_MODEL_PATH = '../yolo_training/runs/exp4-yolo11s/car_detection_combined_dataset/weights/best.pt'
PLATE_MODEL_PATH = '../yolo_training/runs/exp2-plate-yolo11s/plate_detection_from_cars/weights/best.pt'
COLOR_MODEL_PATH = '../color/histogram_experiments/training_results/pca_50/exp_svm_pca_exp_6_bins_L10_A10_B10_H12_S8_V8/svm_model.pkl'
COLOR_ENCODER_PATH = '../color/histogram_experiments/training_results/pca_50/exp_svm_pca_exp_6_bins_L10_A10_B10_H12_S8_V8/label_encoder.pkl'
COLOR_PCA_PATH = '../color/histogram_experiments/training_results/pca_50/exp_svm_pca_exp_6_bins_L10_A10_B10_H12_S8_V8/pca.pkl'
COLOR_SCALER_PATH = '../color/histogram_experiments/training_results/pca_50/exp_svm_pca_exp_6_bins_L10_A10_B10_H12_S8_V8/scaler.pkl'

# ================================
# Cargar modelos
# ================================
car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

color_model = joblib.load(COLOR_MODEL_PATH)
color_encoder = joblib.load(COLOR_ENCODER_PATH)
pca = joblib.load(COLOR_PCA_PATH)
scaler = joblib.load(COLOR_SCALER_PATH)

# ================================
# Lista para guardar resultados
# ================================
results_list = []

class_labels = {
    0: "blanco",
    1: "gris",
    2: "negro",
    3: "rojo",
    4: "amarillo",
    5: "azul",
    6: "verde",
    7: "marrón"
}

def process_frame(frame, frame_id, output_folder, video_name):
    results_car = car_model(frame)
    cars_raw = results_car[0].boxes
    boxes_with_cls = [(box.xyxy[0].cpu().numpy(), int(box.cls[0].cpu().item())) for box in cars_raw]

    filtered_boxes = remove_duplicate_boxes(boxes_with_cls, iou_threshold=0.6)

    plates_dir = Path(output_folder) / "plates"
    plates_dir.mkdir(parents=True, exist_ok=True)

    for i, (box_xyxy, cls_id) in enumerate(filtered_boxes):
        x1, y1, x2, y2 = map(int, box_xyxy)
        car_crop = frame[y1:y2, x1:x2]
        car_type = car_model.names[cls_id]

        # ========== Color ==========
        try:
            features_histogram = extract_features_histogram(
                img=car_crop, bins_L=10, bins_A=10, bins_B=10, bins_H=12, bins_S=8, bins_V=8)
            dominant_color_lab = get_dominant_color_lab(image=car_crop)
            color_features = np.concatenate([features_histogram, dominant_color_lab])
            color_features = scaler.transform([color_features])
            color_features = pca.transform(color_features)
            color_class_idx = color_model.predict(color_features)[0]
            color_name = class_labels.get(color_class_idx, "Unknown")
        except Exception as e:
            color_name = "Unknown"
            print(f"⚠️ Error detectando color: {e}")

        # ========== Detectar placa dentro del crop del carro ==========
        plate_text = None
        best_plate = None
        plate_results = plate_model(car_crop)

        for result in plate_results:
            for box in result.boxes.xyxy:
                best_plate = box.cpu().numpy().astype(int)
                break  # Tomamos la primera placa detectada

        if best_plate is not None:
            px1, py1, px2, py2 = best_plate
            plate_crop = car_crop[py1:py2, px1:px2]

            # Guardar imagen de la placa
            plate_image_path = plates_dir / f"{video_name}_frame{frame_id}_car{i}_plate.jpg"
            cv2.imwrite(str(plate_image_path), plate_crop)

            # OCR
            try:
                image = Image.open(plate_image_path)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image = preprocess_plate_for_ocr(image)
                raw_plate_text = pytesseract.image_to_string(image, config='--oem 3 --psm 7', lang='eng').replace("\f", "").strip()
                plate_text = clean_plate_text(raw_plate_text)
            except Exception as e:
                print(f"⚠️⚠️⚠️ Error OCR placa: {e}")

            if plate_text:
                print(f"✅ Placa detectada: {plate_text}")
            else:
                print("⚠️ No se detectó ninguna placa")
        else:
            print("⚠️⚠️ No se encontró placa en el carro recortado")
            raw_plate_text = ''

        # ========== Guardar imágenes ==========
        cropped_cars_folder = Path(output_folder) / "cropped_cars"
        cropped_cars_folder.mkdir(parents=True, exist_ok=True)
        cropped_image_path = cropped_cars_folder / f"{video_name}_frame{frame_id}_car{i}.jpg"
        cv2.imwrite(str(cropped_image_path), car_crop)

        color_folder = Path(output_folder) / "colors" / color_name
        color_folder.mkdir(parents=True, exist_ok=True)
        image_path = color_folder / f"{video_name}_frame{frame_id}_car{i}.jpg"
        txt_path = color_folder / f"{video_name}_frame{frame_id}_car{i}.txt"

        cv2.imwrite(str(image_path), car_crop)
        with open(txt_path, "w") as f:
            f.write(f"Tipo: {car_type}\n")
            f.write(f"Placa: {plate_text if plate_text else 'No detectada'}\n")

        # ========== Guardar resultados ==========
        results_list.append({
            "video": video_name,
            "frame": frame_id,
            "car_index": i,
            "car_type": car_type,
            "color": color_name,
            "raw_plate_text": raw_plate_text,
            "plate_text": plate_text,
            "image_path": str(image_path)
        })

def process_video(
    video_path,
    output_folder='frames_output',
    csv_output='results.csv',
    results_dir='results'
):
    output_path = Path(results_dir) / output_folder
    csv_path = Path(results_dir) / csv_output
    output_path.mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_name = Path(video_path).stem

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"🎞️ FPS del video: {fps}")
    print(f"🧮 Total de frames: {total_frames}")
    print(f"⏱️ Duración estimada: {duration:.2f} segundos")
    print(f"🔄 Procesando todos los frames...")

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"{video_name}_frame{frame_id}.jpg"
        frame_output_path = output_path / frame_filename
        cv2.imwrite(str(frame_output_path), frame)

        process_frame(frame, frame_id, results_dir, video_name)
        frame_id += 1

    cap.release()

    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    print(f"✅ Video procesado. CSV guardado en: {csv_path}")
    
    summarize_output_path = f'{results_dir}/summarized_plates'
    df = summarize_similar_plates(csv_path, summarize_output_path)


# ================================
# Ejecutar
# ================================
if __name__ == '__main__':
    video_path = '10_min_videos_sin_usar.avi'
    results_dir = '10_min_videos_sin_usar_results' 

    process_video(
    video_path=video_path,
    results_dir=results_dir
    )

