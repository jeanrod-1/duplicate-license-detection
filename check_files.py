import os

label_dir = "all_datasets/cars/labels"

# Revisar archivos que no sean .txt en labels
wrong_files = [f for f in os.listdir(label_dir) if not f.endswith(".txt")]

if wrong_files:
    print("⚠️ Archivos incorrectos en labels:")
    for f in wrong_files:
        print(f)
else:
    print("✅ No hay archivos incorrectos en labels.")
