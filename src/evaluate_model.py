from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import cv2
from src.utils import filter_detections_by_confidence, CLASS_NAMES

# Cargar el modelo YOLO preentrenado
model = YOLO('yolov8m.pt') 

# Función para obtener las predicciones del modelo y convertirlas en clases
def get_predictions(video_path, confidence_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar detección con YOLO
        results = model(frame)

        # Filtrar las detecciones con el umbral de confianza
        filtered_detections = filter_detections_by_confidence(results, confidence_threshold)

        # Obtener las clases detectadas
        for detection in filtered_detections:
            class_id = detection['class_id']
            predictions.append(class_id)

    cap.release()
    return predictions

# Función para evaluar el modelo con métricas
def evaluate_model(video_path, confidence_threshold=0.5):
    # Obtener predicciones del modelo
    predictions = get_predictions(video_path, confidence_threshold)

    # Aquí debes proporcionar las clases verdaderas si tienes datos etiquetados.
    # Para este ejemplo, asumimos que todas las predicciones son correctas (esto debe ajustarse en un caso real).
    true_classes = predictions  # Ajusta esta línea con los datos reales si los tienes.

    # Convertir los IDs de clase a nombres para la visualización
    true_class_names = [CLASS_NAMES.get(cls_id, "Desconocido") for cls_id in true_classes]
    predicted_class_names = [CLASS_NAMES.get(cls_id, "Desconocido") for cls_id in predictions]

    # Calcular las métricas de evaluación
    print("Reporte de clasificación:\n")
    print(classification_report(true_class_names, predicted_class_names, zero_division=0))

    # Generar la matriz de confusión
    conf_matrix = confusion_matrix(true_class_names, predicted_class_names, labels=list(CLASS_NAMES.values()))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=list(CLASS_NAMES.values()),
        yticklabels=list(CLASS_NAMES.values())
    )
    plt.xlabel('Predicciones')
    plt.ylabel('Clases Verdaderas')
    plt.title('Matriz de Confusión')
    plt.show()

# Evaluar el modelo
evaluate_model('data/video.mp4', confidence_threshold=0.5)
