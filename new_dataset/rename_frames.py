import os

# Definir la carpeta de imágenes y el prefijo
image_folder = "extracted_frames"
prefix = "2025-02-26_puente_127"

# Obtener la lista de archivos en la carpeta
for filename in os.listdir(image_folder):
    old_path = os.path.join(image_folder, filename)

    # Verificar si es un archivo de imagen (JPG)
    if filename.endswith(".jpg"):
        new_filename = prefix + filename  # Agregar prefijo
        new_path = os.path.join(image_folder, new_filename)
        
        # Renombrar archivo
        os.rename(old_path, new_path)
        print(f"Renombrado: {filename} → {new_filename}")

print("Renombrado completado 🚀")
