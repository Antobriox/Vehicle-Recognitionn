import os
from src.core.detector import VehicleDetector
from src.utils.constants import INPUT_PATH, OUTPUT_PATH, REPORTS_PATH

def main():
    # Crear directorios necesarios
    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)
    
    # Verificar que existe el video
    input_video = os.path.join(INPUT_PATH, 'video.mp4')
    if not os.path.exists(input_video):
        print(f"Error: No se encontró el video en {input_video}")
        print("Por favor, coloca un video en la carpeta data/input/ con el nombre 'video.mp4'")
        return
    
    # Inicializar detector
    detector = VehicleDetector()
    output_video = os.path.join(OUTPUT_PATH, 'output.avi')
    
    try:
        detector.process_video(
            input_video,
            output_video,
            confidence_threshold=0.5,
            generate_report=True
        )
        print(f"Video procesado guardado en: {output_video}")
        
    except Exception as e:
        print(f"Error al procesar el video: {e}")

if __name__ == "__main__":
    main() 