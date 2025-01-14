# Esto permite importar las funciones principales de los módulos sin necesidad de hacer referencia al módulo explícitamente.
from .detect_vehicles import process_video_yolo, process_video_with_filter
from .evaluate_model import evaluate_model
from .utils import (
    load_video,
    save_processed_video,
    preprocess_image,
    show_video_in_real_time,
    filter_detections_by_confidence,
    show_predictions,
)
from .tracker import track_objects
