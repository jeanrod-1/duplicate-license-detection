import os
import cv2
import matplotlib.pyplot as plt

# Directorio base del nuevo dataset
dataset_path = "Dataset_completo"
output_dir = os.path.join("ejemplo_labels_dataset_completo")  # Carpeta donde guardaremos los ejemplos

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Diccionario para almacenar un ejemplo por clase
saved_classes = {}

# Recorrer los videos dentro del dataset
for video_folder in os.listdir(dataset_path):
    video_path = os.path.join(dataset_path, video_folder)
    
    if not os.path.isdir(video_path):  # Saltar si no es un directorio
        continue

    # Buscar imágenes y etiquetas dentro de cada video
    for file in os.listdir(video_path):
        if file.endswith(".txt"):  # Buscar archivos de etiquetas
            label_path = os.path.join(video_path, file)
            image_name = file.replace(".txt", ".jpg")  # Cambiar sufijo para encontrar la imagen
            image_path = os.path.join(video_path, image_name)

            if not os.path.exists(image_path):  # Saltar si no existe la imagen
                continue

            # Cargar imagen
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape

            # Leer archivo de etiquetas
            with open(label_path, "r") as f:
                for line in f.readlines():
                    data = line.strip().split()
                    class_id = int(data[0])

                    # Si ya guardamos un ejemplo de esta clase, saltar
                    if class_id in saved_classes:
                        continue

                    # Extraer coordenadas normalizadas y convertirlas a píxeles
                    x_center, y_center, w, h = map(float, data[1:])
                    x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
                    x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                    x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

                    # Dibujar bounding box y etiqueta
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, f"Class {class_id}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Guardar la imagen en el directorio de ejemplo
                    output_path = os.path.join(output_dir, f"class_{class_id}.jpg")
                    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    saved_classes[class_id] = image_name  # Marcar la clase como guardada

                    print(f"Guardado ejemplo de clase {class_id} en {output_path}")

                    # Una vez guardado un ejemplo de esta clase, salir del bucle
                    break