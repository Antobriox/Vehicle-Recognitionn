# Proyecto de Detección y Seguimiento de Vehículos

Este proyecto utiliza YOLOv8 para detectar y rastrear vehículos como carros, motos, buses y camionetas en videos. Además, implementa un sistema de seguimiento de objetos para asignar ID únicos a los vehículos detectados.

## Características
- Detección precisa de vehículos usando YOLOv8.
- Rastreo de vehículos con IDs únicos utilizando el algoritmo SORT.


## Requisitos

### Dependencias
Las siguientes dependencias son necesarias para ejecutar el proyecto:

- Python 3.8 o superior
- ultralytics
- numpy
- opencv-python
- tensorflow
- matplotlib
- scikit-learn
- seaborn

Puedes instalar todas las dependencias usando el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Uso

### Ejecución del Proyecto
1. Coloca el video que deseas procesar en la carpeta `data` y asegúrate de que se llame `video.mp4` (o ajusta el nombre en el código si es diferente).
2. Ejecuta el archivo `detect_vehicles.py`:

```bash
python -m src.detect_vehicles
```

3. El video procesado se guardará en `data/output.avi`.

### Evaluación del Modelo
Para evaluar el rendimiento del modelo, puedes ejecutar el archivo `evaluate_model.py`:

```bash
python -m src.evaluate_model
```

Esto generará un reporte de clasificación y una matriz de confusión.

## Personalización
- Puedes ajustar los umbrales de confianza y los colores de las clases en los archivos correspondientes (`utils.py` y `detect_vehicles.py`).
- El modelo YOLO utilizado es `yolov8m.pt`, pero puedes cambiarlo por otro (como `yolov8n.pt`) según tus necesidades.



