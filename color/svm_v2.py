import pickle
import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# 🔹 Configuración del experimento
# -------------------------------
experiment_name = "exp9-svm-new-features-old-dataset"
base_output_dir = "training_results"
output_dir = os.path.join(base_output_dir, experiment_name)
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 🔹 Cargar características y etiquetas
# -------------------------------
features_file = "features_output/image_features_v2.pkl"
with open(features_file, "rb") as f:
    features_dict = pickle.load(f)

# Obtener el primer elemento del diccionario
first_key = next(iter(features_dict))  # Obtiene la primera clave (nombre de la imagen)
first_feature = features_dict[first_key]  # Obtiene las características asociadas

# Mostrar la dimensionalidad
print(f"Dimensión de las características de '{first_key}': {first_feature.shape}")

# Leer el archivo de etiquetas
labels_path = "cropped_cars_v2/labels.txt"

with open(labels_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Procesar las etiquetas correctamente
data = []
for line in lines:
    parts = line.strip().split(" ")
    image_name = parts[0]  # El nombre de la imagen está en la primera posición
    label = int(parts[1])  # Convertir el label a entero
    data.append((image_name, label))

df = pd.DataFrame(data, columns=["image", "label"])
X = np.array([features_dict[img] for img in df["image"]])

# Codificar etiquetas
le = LabelEncoder()
y = le.fit_transform(df["label"])
num_classes = len(le.classes_)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Normalizar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 🔹 Definir la grilla de hiperparámetros
# -------------------------------
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# -------------------------------
# 🔹 Entrenamiento con GridSearchCV
# -------------------------------
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Obtener mejores hiperparámetros
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluar el modelo
y_pred = best_model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, normalize='true')

# Guardar resultados
with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
    json.dump(best_params, f, indent=4)

class_names = le.classes_

# Graficar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

import joblib

# Guarda el modelo SVM entrenado
joblib.dump(best_model, os.path.join(output_dir, "svm_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))



print(f"Resultados guardados en {output_dir}/")
