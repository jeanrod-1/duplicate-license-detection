import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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
experiment_name = "exp5-nn" 
base_output_dir = "training_results"
output_dir = os.path.join(base_output_dir, experiment_name)
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 🔹 Cargar características y etiquetas
# -------------------------------
features_file = "features_output/image_features.pkl"
with open(features_file, "rb") as f:
    features_dict = pickle.load(f)

labels_path = "labels.txt"

with open(labels_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Separar la última palabra como label y el resto como nombre de imagen
data = []
for line in lines:
    parts = line.strip().split(" ")  # Dividir por espacios
    image_name = " ".join(parts[:-1])  # Todo menos la última palabra es el nombre de la imagen
    label = parts[-1]  # Última palabra es el label
    data.append((image_name, label))

# Convertir a DataFrame
df = pd.DataFrame(data, columns=["image", "label"])

# Convertir las features a un array
X = np.array([features_dict[img] for img in df["image"]])

# Codificar etiquetas a números
le = LabelEncoder()
y = le.fit_transform(df["label"])
num_classes = len(le.classes_)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Obtener los valores únicos de las clases
class_labels = np.unique(y)

# Calcular los pesos de las clases
class_weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=y)

# Convertir a diccionario para pasarlo a `model.fit()`
class_weight_dict = dict(zip(class_labels, class_weights))

# -------------------------------
# 🔹 Definir los Hiperparámetros
# -------------------------------
hyperparams = {
    "input_dim": X.shape[1],
    "hidden_layers": [
        {"units": 256, "activation": "relu", "dropout": 0.3},
        {"units": 128, "activation": "relu", "dropout": 0.2},
        {"units": 32, "activation": "relu", "dropout": 0.2}
    ],
    "output_classes": num_classes,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss_function": "categorical_crossentropy",
    "batch_size": 64,
    "epochs": 150
}

# -------------------------------
# 🔹 Construcción del Modelo
# -------------------------------
model = Sequential()
model.add(Dense(hyperparams["hidden_layers"][0]["units"], activation=hyperparams["hidden_layers"][0]["activation"], input_shape=(X.shape[1],)))
model.add(Dropout(hyperparams["hidden_layers"][0]["dropout"]))
model.add(Dense(hyperparams["hidden_layers"][1]["units"], activation=hyperparams["hidden_layers"][1]["activation"]))
model.add(Dropout(hyperparams["hidden_layers"][1]["dropout"]))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(hyperparams["hidden_layers"][2]["units"], activation=hyperparams["hidden_layers"][2]["activation"]))
model.add(Dropout(hyperparams["hidden_layers"][2]["dropout"]))
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=hyperparams["learning_rate"]), 
                loss=hyperparams["loss_function"], 
                metrics=["accuracy"]
                )

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

# -------------------------------
# 🔹 Entrenar el Modelo
# -------------------------------
history = model.fit(X_train, y_train_cat, 
                    validation_data=(X_test, y_test_cat),
                    epochs=hyperparams["epochs"], 
                    batch_size=hyperparams["batch_size"],
                    class_weight=class_weight_dict,
                    callbacks=[early_stop]
                    )

# Evaluar el modelo
loss, acc = model.evaluate(X_test, y_test_cat)

# Predicciones en test
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# -------------------------------
# 🔹 Guardar Resultados y Métricas
# -------------------------------

# Guardar clasificación
report = classification_report(y_test, y_pred_classes, target_names=le.classes_, output_dict=True)
with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

# Guardar métricas en un archivo txt
metrics_file = os.path.join(output_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Accuracy en test: {acc:.4f}\n")
    f.write(f"Loss en test: {loss:.4f}\n\n")
    f.write("Precision, Recall y F1-score por clase:\n")
    for label, values in report.items():
        if isinstance(values, dict):
            f.write(f"{label}: Precision={values['precision']:.4f}, Recall={values['recall']:.4f}, F1-score={values['f1-score']:.4f}\n")
    f.write(f"\nMacro F1-score: {report['macro avg']['f1-score']:.4f}\n")
    f.write(f"Weighted F1-score: {report['weighted avg']['f1-score']:.4f}\n")

# Guardar historial de entrenamiento en un archivo JSON
history_file = os.path.join(output_dir, "training_history.json")
with open(history_file, "w") as f:
    json.dump(history.history, f)

# Guardar hiperparámetros en un archivo JSON
hyperparams_file = os.path.join(output_dir, "hyperparameters.json")
with open(hyperparams_file, "w") as f:
    json.dump(hyperparams, f, indent=4)

# -------------------------------
# 🔹 Graficar Resultados y Guardar
# -------------------------------
# Curva de precisión
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()
plt.title("Curva de Precisión del Modelo")
accuracy_plot = os.path.join(output_dir, "training_accuracy.png")
plt.savefig(accuracy_plot)
plt.close()

# Graficar evolución del loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Evolución del Loss')
plt.savefig(os.path.join(output_dir, "loss_evolution.png"))
plt.close()

# Matriz de confusión
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred_classes, normalize='true')

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# Guardar el modelo entrenado
model_file = os.path.join(output_dir, "model.h5")
model.save(model_file)

print(f"Resultados guardados en {output_dir}/")
