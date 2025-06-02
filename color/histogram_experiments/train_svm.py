import pickle
import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------
# 🔹 Configuración del experimento
# -------------------------------
base_output_dir = "training_results/pca_200"
os.makedirs(base_output_dir, exist_ok=True)

# Obtener lista de archivos de características
features_dir = "./features_output/"
feature_files = [f for f in os.listdir(features_dir) if f.endswith(".pkl")]

# Mapeo de etiquetas
class_labels = {
    0: "blanco",
    1: "gris",
    2: "negro",
    3: "rojo",
    4: "amarillo",
    5: "azul",
    6: "verde",
    7: "marrón"
}

for i, features_file in enumerate(feature_files, 1):
    print(f"🔄 Ejecutando experimento {i}...")
    experiment_name = f"exp_svm_pca_{os.path.splitext(features_file)[0]}"
    output_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # 🔹 Cargar características y etiquetas
    # -------------------------------
    with open(os.path.join(features_dir, features_file), "rb") as f:
        features_dict = pickle.load(f)

    first_key = next(iter(features_dict))
    first_feature = features_dict[first_key]
    print(f"Dimensión de las características de '{first_key}': {first_feature.shape}")

    labels_path = "../cropped_cars_v2/labels.txt"
    with open(labels_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        parts = line.strip().split(" ")
        image_name = parts[0]
        label = int(parts[1])
        data.append((image_name, label))

    df = pd.DataFrame(data, columns=["image", "label"])
    X = np.array([features_dict[img] for img in df["image"]])

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # 🔹 Aplicar PCA
    # -------------------------------
    # pca = PCA(n_components='mle', svd_solver='full')
    pca = PCA(n_components=200)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

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

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    # Guardar resultados
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # Graficar matriz de confusión con etiquetas
    labels = [class_labels[i] for i in range(len(class_labels))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Guardar modelos
    joblib.dump(best_model, os.path.join(output_dir, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))
    joblib.dump(pca, os.path.join(output_dir, "pca.pkl"))

    print(f"Resultados guardados en {output_dir}/")

print("✅ Todos los experimentos completados.")
