import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tracker import track_objects
from ultralytics import YOLO
import cv2
from src.utils import filter_detections_by_confidence, CLASS_NAMES

# Diccionario de colores para cada tipo de vehículo
CLASS_COLORS = {
    2: (255, 255, 255),  
    3: (0, 255, 0),      
    5: (255, 255, 0),    
    7: (255, 0, 0),      
}

# Cargar el modelo YOLO preentrenado
model = YOLO('yolov8m.pt')  

# Función para procesar el video cuadro por cuadro y realizar la detección
def process_video_yolo(video_path, output_path, confidence_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar detección con YOLO
        results = model(frame)

        # Filtrar las detecciones por confianza
        filtered_detections = filter_detections_by_confidence(results, confidence_threshold)

        # Preparar las detecciones para el rastreador
        detections_for_tracker = [
            [det['coordinates'][0], det['coordinates'][1], det['coordinates'][2], det['coordinates'][3], det['confidence']]
            for det in filtered_detections
        ]

        # Aplicar seguimiento de objetos
        tracked_objects = track_objects(detections_for_tracker)

        # Dibujar los cuadros y etiquetas en el marco
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            for det in filtered_detections:
                if abs(det['coordinates'][0] - x1) < 10 and abs(det['coordinates'][1] - y1) < 10:
                    class_id = det['class_id']
                    class_name = CLASS_NAMES.get(class_id, "Desconocido")

                    # Obtener el color basado en la clase
                    color = CLASS_COLORS.get(class_id, (0, 0, 255))  # Rojo predeterminado si la clase no está en el diccionario

                    # Dibujar el cuadro delimitador con el color correspondiente
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Dibujar la etiqueta con el nombre de la clase y el ID del objeto
                    cv2.putText(
                        frame,
                        f"{class_name} ID: {int(track_id)}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

        # Guardar el cuadro procesado
        out.write(frame)

        # Mostrar el video en tiempo real 
        cv2.imshow('Seguimiento', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Llamar a la función con el video
process_video_yolo('data/video.mp4', 'data/output.avi', confidence_threshold=0.5)
