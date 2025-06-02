import os
import shutil
import re
from collections import Counter

def get_existing_filenames(base_dirs):
    """Recorre los directorios y obtiene solo los nombres de los archivos (sin rutas)."""
    existing_files = set()
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                existing_files.add(file)  # Solo guarda el nombre del archivo, sin la ruta
    return existing_files

def extract_video_name(image_name):
    """Extrae el nombre del video basado en su presencia en el nombre del archivo."""
    match = re.match(r"(.*? \d{2}_\d{2}_\d{2})", image_name)
    if match:
        return match.group(1)
    return re.sub(r"(_\d+)?\.(jpg|jpeg|png|txt|json)$", "", image_name)

def count_images_by_video(files):
    """Cuenta la cantidad de imágenes por video en un conjunto de archivos, considerando solo extensiones de imagen."""
    counter = Counter()
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png")):
            # print(f"Procesando archivo: {file}")  # Agregado para depuración
            video_name = extract_video_name(file)
            if video_name and video_name in file:
                counter[video_name] += 1
    return counter

def copy_missing_images(source_dirs, all_datasets_dirs, missing_images_dir, missing_dataset_dir):
    """Copia imágenes que no están en all_datasets ni en missing_images, asegurando que el .txt asociado exista."""
    os.makedirs(missing_images_dir, exist_ok=True)
    
    # Obtener nombres de archivos existentes en all_datasets y missing_images
    existing_files = get_existing_filenames(all_datasets_dirs)
    missing_files = get_existing_filenames([missing_images_dir])
    missing_files_datasets = get_existing_filenames([missing_dataset_dir])
    printed_videos = set()
    
    # existing_counts = count_images_by_video(existing_files)

    copied_count = 0  # Contador de imágenes copiadas
    
    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            # new_files = {os.path.relpath(os.path.join(root, file), source_dir) for file in files}
            new_files = {os.path.relpath(os.path.join(root, file), root) for file in files}
            new_counts = count_images_by_video(new_files)
            
            for file in new_files:
                if file.endswith((".mp4", ".avi", ".json", ".names")):
                    continue  # Ignorar videos y archivos JSON
                
                if file.endswith(".txt") and file in existing_files:
                    continue  # Ignorar etiquetas si ya están en labels
                
                if file in existing_files or file in missing_files or file in missing_files_datasets:
                    continue  # Si ya existe en all_datasets o missing_images, omitir
                
                # Verificar si existe el archivo .txt correspondiente antes de copiar la imagen
                if file.endswith((".jpg", ".jpeg", ".png")):
                    label_file = re.sub(r"\.(jpg|jpeg|png)$", ".txt", file)
                    label_source_path = os.path.join(root, label_file)  # ✅ Mantiene la estructura correcta
                    # print(label_source_path)
                    if os.path.exists(label_source_path):
                        source_path = os.path.join(root, file)
                        dest_path = os.path.join(missing_images_dir, file)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(source_path, dest_path)
                        copied_count += 1  # Incrementar contador
                        print(f"✅ Copiada nueva imagen: {file}")
                        
                        # Copiar la etiqueta correspondiente
                        label_dest_path = os.path.join(missing_images_dir, label_file)
                        shutil.copy2(label_source_path, label_dest_path)
                        # print(f"✅ Copiada etiqueta asociada: {label_file}")

    print(f"\n🔹 Total de imágenes copiadas de {source_dir}: {copied_count}")

if __name__ == "__main__":
    all_datasets_dirs = [
        "all_datasets"
    ]
    missing_images_dir = "all_missing_images"
    missing_dataset_dir = "all_missing_datasets"
    new_source_dirs = ["/HDDmedia/gemeleados/Prueba_V1/Pruebita/video_import_2023-02-diego500"]
    
    copy_missing_images(new_source_dirs, all_datasets_dirs, missing_images_dir, missing_dataset_dir)
