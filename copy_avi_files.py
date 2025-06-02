import os
import shutil
import zipfile

def copy_unique_avi_files(source_dir, destination_folder):  
    destination_path = os.path.join(os.getcwd(), destination_folder)
    os.makedirs(destination_path, exist_ok=True)
    
    seen_files = set()
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".avi") and file not in seen_files:
                seen_files.add(file)
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_path, file)
                
                shutil.copy2(source_file_path, destination_file_path)
                print(f"Copiado: {file}")


def check_images_for_videos(destination_folder, dataset_folder):
    destination_path = os.path.join(os.getcwd(), destination_folder)
    dataset_path = os.path.join(os.getcwd(), dataset_folder)
    
    videos_without_images = []
    
    for video_file in os.listdir(destination_path):
        if video_file.lower().endswith(".avi"):
            base_name = video_file.rsplit(".", 1)[0]  # Remove .avi extension
            found = False
            
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if base_name in file:
                        # print(f"🖼️ Imagen encontrada para el video {video_file}: {file} en {root}")
                        found = True
                        break
                if found:
                    break
            
            if not found:
                print(f"❌ No se encontró imagen para el video {video_file}")
                videos_without_images.append(os.path.join(destination_path, video_file))
    
    if videos_without_images:
        zip_videos_without_images(videos_without_images, "videos_sin_usar.zip")

def zip_videos_without_images(videos, zip_name):
    zip_path = os.path.join(os.getcwd(), zip_name)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for video in videos:
            zipf.write(video, os.path.basename(video))
    
    print(f"📦 Archivo ZIP creado: {zip_name}")

# Uso
source_directory = "/HDDmedia/gemeleados"  
destination_folder = "full_videos"
dataset_directory = "all_datasets"  
# copy_unique_avi_files(source_directory, destination_folder)
check_images_for_videos(destination_folder, dataset_directory)
