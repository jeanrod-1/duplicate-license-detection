import os
import shutil

# Ruta del dataset
dataset_path = "Dataset_completo"

# Recorrer los directorios dentro de dataset2
for video_folder in os.listdir(dataset_path):
    video_path = os.path.join(dataset_path, video_folder)

    if not os.path.isdir(video_path):  # Saltar si no es un directorio
        continue

    # Verificar si contiene solo una subcarpeta
    subdirs = os.listdir(video_path)
    for subdir in subdirs:
        subfolder_path = os.path.join(video_path, subdir)

        if os.path.isdir(subfolder_path):  # Asegurar que sea un directorio
            new_location = os.path.join(dataset_path, subdirs[0])
            
            # Mover la carpeta completa al nivel superior
            shutil.move(subfolder_path, new_location)
            print(f"📂 Movido: {subfolder_path} → {new_location}")

            # Eliminar la carpeta original vacía
            # os.rmdir(video_path)
            print(f"🗑 Eliminado: {video_path}")

print("✅ Reorganización completada.")
