import os
import pandas as pd
from collections import Counter

# Ruta de la carpeta con los archivos TXT
carpeta = "/data/estudiantes/gemeleados/rodriguezjean/dataset_cars/labels/train"

# Contador para los labels
contador_labels = Counter()

# Leer todos los archivos TXT en la carpeta
for archivo in os.listdir(carpeta):
    if archivo.endswith(".txt"):
        ruta_archivo = os.path.join(carpeta, archivo)

        # Verificar si el archivo no está vacío antes de leerlo
        if os.path.getsize(ruta_archivo) > 0:
            df = pd.read_csv(ruta_archivo, delim_whitespace=True, header=None, usecols=[0])
            contador_labels.update(df[0])

# Convertir el contador a un DataFrame ordenado
df_counts = pd.DataFrame(contador_labels.items(), columns=["Label", "Frecuencia"]).sort_values(by="Frecuencia", ascending=False)

# Mostrar el resultado
print(df_counts)
print(df_counts.sum())
