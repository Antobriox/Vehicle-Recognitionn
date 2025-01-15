import cv2
import numpy as np

# Diccionario de nombres de clases (según el COCO Dataset)
CLASS_NAMES = {
    2: "Car",      
    3: "Motorcycle",  
    5: "Bus",       
    7: "Truck"      
}

# Filtrar detecciones por confianza
def filter_detections_by_confidence(results, confidence_threshold=0.3):
    """
    Filtra las detecciones del modelo en función del umbral de confianza.
    results: Resultados obtenidos del modelo YOLO.
    confidence_threshold: Umbral de confianza predeterminado para filtrar las detecciones.
    """
    filtered_detections = []
    for result in results[0].boxes:
        confidence = float(result.conf)  # Confianza de la detección
        class_id = int(result.cls)  # ID de la clase detectada

        # Ajustar el umbral según la clase
        if (class_id == 3 and confidence >= 0.2) or confidence >= confidence_threshold:
            coordinates = result.xywh[0]
            x, y, w, h = float(coordinates[0]), float(coordinates[1]), float(coordinates[2]), float(coordinates[3])
            x1 = x - w / 2  # Coordenada X superior izquierda
            y1 = y - h / 2  # Coordenada Y superior izquierda
            x2 = x + w / 2  # Coordenada X inferior derecha
            y2 = y + h / 2  # Coordenada Y inferior derecha
            filtered_detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'coordinates': (x1, y1, x2, y2)
            })

    return filtered_detections
