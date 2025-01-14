import numpy as np
from src.sort import Sort  

# Inicializar el rastreador
tracker = Sort(max_age=10, min_hits=3)  # Ajusta los parámetros según sea necesario

# Función para realizar el seguimiento
def track_objects(detections):
    """
    Realiza el seguimiento de los objetos detectados.
    detections: lista de detecciones en formato [x1, y1, x2, y2, score]
    """
    # Actualizar el rastreador con las detecciones actuales
    tracks = tracker.update(np.array(detections))  # Actualiza las detecciones y devuelve los objetos rastreados

    # tracks tendrá la forma [x1, y1, x2, y2, id]
    results = []
    for track in tracks:
        x1, y1, x2, y2, track_id = track  # Extrae las coordenadas y el ID
        results.append((x1, y1, x2, y2, int(track_id)))  # Asegúrate de convertir el ID a entero

    return results  # Devuelve los objetos con sus IDs
