import cv2
from .constants import CLASS_NAMES, CLASS_COLORS

def draw_detection(frame, track, detection):
    """
    Dibuja la detección en el frame con el ID de tracking.
    """
    try:
        x1, y1, x2, y2, track_id = track
        class_id = detection['class_id']
        
        # Obtener nombre de clase y color
        class_name = CLASS_NAMES.get(class_id, "Desconocido")
        color = CLASS_COLORS.get(class_id, (0, 0, 255))
        
        # Dibujar bbox
        cv2.rectangle(frame, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     color, 
                     2)
        
        # Dibujar etiqueta
        label = f"{class_name} ID:{int(track_id)}"
        cv2.putText(frame,
                   label,
                   (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   color,
                   2)
                   
    except Exception as e:
        print(f"Error dibujando detección: {e}") 