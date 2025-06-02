import os

# Definir las rutas
image_folder = "extracted_frames"
train_txt_path = "data/train.txt"

# Obtener los nombres de los frames en extracted_frames (estructura: 2025-02-26_puente_127_frame_000017.jpg)
extracted_frames = set()
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):  # Solo imágenes
        extracted_frames.add(filename.split("frame_")[-1].replace(".jpg", ""))

# Obtener los nombres de los frames en train.txt (estructura: data/images/train/frame_000017.PNG)
train_frames = set()
with open(train_txt_path, "r") as f:
    for line in f:
        if "/frame_" in line:
            frame_name = line.strip().split("/")[-1].replace(".PNG", "").replace("frame_", "")
            train_frames.add(frame_name)

# Comparar los conjuntos
missing_frames = train_frames - extracted_frames  # En train.txt pero NO en extracted_frames

# Imprimir resultados
print(f"📂 Imágenes en 'extracted_frames/': {len(extracted_frames)}")
print(f"📜 Frames en 'train.txt': {len(train_frames)}")
print(f"⚠️ Frames en train.txt pero NO en extracted_frames: {len(missing_frames)}")

# Mostrar los primeros 10 frames faltantes (para revisión)
if missing_frames:
    print("Ejemplo de frames faltantes:", sorted(missing_frames)[:10])
else:
    print("✅ Todos los frames de train.txt están en extracted_frames.")
