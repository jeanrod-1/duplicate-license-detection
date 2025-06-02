import os
import collections

# 📂 Directorio raíz donde están los datasets separados
dataset_separado = "all_datasets"

# 📂 Subdirectorios dentro de all_datasets
datasets = ["cars", "plates", "plates_numbers_and_letters"]

# 🔄 Recorrer cada dataset y contar imágenes + etiquetas
for dataset in datasets:
    img_dir = os.path.join(dataset_separado, dataset, "images")
    label_dir = os.path.join(dataset_separado, dataset, "labels")

    # 📊 Contar imágenes
    num_images = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    # 📊 Contar clases en etiquetas
    class_counts = collections.Counter()

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)

            with open(label_path, "r") as file:
                for line in file:
                    class_id = line.split()[0]  # Extraer solo la clase
                    class_counts[class_id] += 1

    # 📢 Mostrar resultados
    print(f"\n📌 Dataset: {dataset}")
    print(f"   📷 Total imágenes: {num_images}")
    print("   🏷️ Conteo de clases:")
    for class_id, count in sorted(class_counts.items(), key=lambda x: int(x[0])):
        print(f"      - Clase {class_id}: {count} etiquetas")

print("\n✅ Conteo completado.")
