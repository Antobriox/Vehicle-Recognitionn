from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import cv2
import os
from ..utils.constants import CLASS_NAMES, MODEL_PATH, INPUT_PATH
from ..utils.detection_utils import filter_detections_by_confidence

def evaluate_model(video_path=None, confidence_threshold=0.5):
    if video_path is None:
        video_path = os.path.join(INPUT_PATH, 'video.mp4')
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encontró el video en: {video_path}")
    
    # Cargar el modelo YOLO preentrenado
    model = YOLO(MODEL_PATH)
    predictions = []
    
    print(f"Analizando video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:  # Procesar cada 10 frames para acelerar
            results = model(frame)
            filtered_detections = filter_detections_by_confidence(results, confidence_threshold)
            
            for detection in filtered_detections:
                class_id = detection['class_id']
                if class_id in CLASS_NAMES:  # Solo agregar clases válidas
                    predictions.append(class_id)
    
    cap.release()
    
    if not predictions:
        print("No se detectaron vehículos en el video.")
        return
    
    # Usar las predicciones como ground truth para este ejemplo
    true_classes = predictions
    
    true_class_names = [CLASS_NAMES[cls_id] for cls_id in true_classes if cls_id in CLASS_NAMES]
    predicted_class_names = [CLASS_NAMES[cls_id] for cls_id in predictions if cls_id in CLASS_NAMES]
    
    if not true_class_names:
        print("No se encontraron clases válidas para evaluar.")
        return
    
    print("\nReporte de clasificación:")
    print(classification_report(true_class_names, predicted_class_names, zero_division=0))
    
    # Matriz de confusión
    unique_classes = sorted(set(CLASS_NAMES.values()))
    conf_matrix = confusion_matrix(
        true_class_names, 
        predicted_class_names, 
        labels=unique_classes
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=unique_classes,
        yticklabels=unique_classes
    )
    plt.xlabel('Predicciones')
    plt.ylabel('Clases Verdaderas')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"Error durante la evaluación: {e}")
