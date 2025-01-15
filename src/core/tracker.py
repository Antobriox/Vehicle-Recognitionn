import numpy as np
from .sort import Sort  

# Inicializar el rastreador
tracker = Sort(max_age=10, min_hits=3)

def track_objects(detections):
    """
    Realiza el seguimiento de los objetos detectados.
    detections: lista de detecciones en formato [x1, y1, x2, y2, score]
    """
    # Actualizar el rastreador con las detecciones actuales
    tracks = tracker.update(np.array(detections))

    # tracks tendrá la forma [x1, y1, x2, y2, id]
    results = []
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        results.append((x1, y1, x2, y2, int(track_id)))

    return results 