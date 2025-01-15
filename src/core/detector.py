from ultralytics import YOLO
import cv2
import os
from ..utils.constants import CLASS_NAMES, CLASS_COLORS, MODEL_PATH
from .tracker import track_objects
from .analyzer import VehicleAnalyzer
from ..utils.visualization import draw_detection
from ..utils.detection_utils import filter_detections_by_confidence

class VehicleDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.analyzer = VehicleAnalyzer()
    
    def process_video(self, input_path, output_path, confidence_threshold=0.5, generate_report=True):
        if not os.path.exists(input_path):
            raise ValueError(f"No se encontró el video en: {input_path}")
            
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {input_path}")
        
        # Configurar video de salida
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Detectar vehículos
                results = self.model(frame)
                detections = filter_detections_by_confidence(results, confidence_threshold)
                
                if detections:  # Solo procesar si hay detecciones
                    # Preparar detecciones para el tracker
                    detections_for_tracker = [
                        [det['coordinates'][0], det['coordinates'][1],
                         det['coordinates'][2], det['coordinates'][3],
                         det['confidence']]
                        for det in detections
                    ]
                    
                    # Rastrear objetos
                    tracked_objects = track_objects(detections_for_tracker)
                    
                    # Procesar y visualizar detecciones
                    for track in tracked_objects:
                        x1, y1, x2, y2, track_id = track
                        for det in detections:
                            if abs(det['coordinates'][0] - x1) < 10 and abs(det['coordinates'][1] - y1) < 10:
                                # Visualizar detección
                                draw_detection(frame, track, det)
                                
                                # Agregar al analizador
                                self.analyzer.add_detection(
                                    frame_count,
                                    det['class_id'],
                                    CLASS_NAMES[det['class_id']],
                                    det['confidence'],
                                    track_id
                                )
                
                out.write(frame)
                cv2.imshow('Detección de Vehículos', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error procesando frame {frame_count}: {e}")
            raise e
            
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            if generate_report:
                stats = self.analyzer.analyze_detections()
                self.analyzer.generate_pdf_report(stats)
