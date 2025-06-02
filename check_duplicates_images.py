import os

def get_existing_filenames(base_dirs):
    """Recorre los directorios y obtiene solo los nombres de los archivos (sin rutas)."""
    existing_files = set()
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                existing_files.add(file)  # Solo guarda el nombre del archivo, sin la ruta
    return existing_files


def check_duplicate_images(missing_dir, all_datasets_dir):
    """Verifica si alguna imagen en missing_images ya está en all_datasets."""
    missing_files = get_existing_filenames([missing_dir])
    all_dataset_files = get_existing_filenames([all_datasets_dir])

    duplicates = missing_files.intersection(all_dataset_files)

    if duplicates:
        print("⚠️ ¡Se encontraron imágenes duplicadas en all_datasets! ⚠️")
        for dup in duplicates:
            print(f"📌 {dup}")
    else:
        print("✅ No hay imágenes duplicadas. Todo está correcto.")

if __name__ == "__main__":
    missing_images_dir = "missing_images"
    all_datasets_dir = "all_datasets"
    # print(get_existing_filenames([all_datasets_dir]))
    # 'HTM630-C31-V-CR7-CL45-CHAPINERO-31-07-2019 13_23_56_frame_000042.jpg'
    # print(get_existing_filenames([missing_images_dir]))
    # 'NCZ418-C31-V-CR7-CL45-CHAPINERO-23-07-2019 16_43_12_frame_000002.jpg'

    check_duplicate_images(missing_images_dir, all_datasets_dir)
