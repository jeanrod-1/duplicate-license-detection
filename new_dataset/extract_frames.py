import cv2
import os

# Rutas de entrada y salida
train_txt_path = "data/train.txt"  # Ruta del archivo train.txt extraído
video_file = "raw/17min_puente_127.avi"
output_folder = "extracted_frames"

# Leer los nombres de los frames desde train.txt
with open(train_txt_path, "r") as f:
    frame_names = {os.path.basename(line.strip()) for line in f}

# Crear la carpeta de salida
os.makedirs(output_folder, exist_ok=True)

# Abrir el video con OpenCV
cap = cv2.VideoCapture(video_file)
frame_index = 0
extracted_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Salir cuando se acaben los frames
    
    frame_name = f"frame_{frame_index:06d}.PNG"  # Se mantiene la estructura para la comparación

    if frame_name in frame_names:
        output_path = os.path.join(output_folder, f"frame_{frame_index:06d}.jpg")  # Se guarda como JPG
        cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        extracted_count += 1

    # Mostrar progreso cada 100 frames
    if frame_index % 100 == 0:
        print(f"Procesados {frame_index} frames... ({extracted_count} extraídos)")

    frame_index += 1  # Ahora el incremento está fuera del if

cap.release()
cv2.destroyAllWindows()

print(f"Extracción completada. {extracted_count} frames guardados en '{output_folder}'")