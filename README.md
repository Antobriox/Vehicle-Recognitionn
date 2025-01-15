# Reconocimiento de Vehículos en Video Usando Inteligencia Artificial

Este proyecto utiliza YOLOv8 para detectar y rastrear vehículos como autos, motos, buses y camionetas en videos. Además, implementa un sistema de seguimiento de objetos para asignar IDs únicos a los vehículos detectados y generar reportes detallados con métricas de rendimiento.

## Características

- **Detección Precisa de Vehículos**: Utiliza YOLOv8 para identificar vehículos en tiempo real.
- **Rastreo de Objetos**: Implementa el algoritmo SORT para asignar IDs únicos y seguir los vehículos a lo largo del video.
- **Análisis y Reportes**: Genera análisis estadísticos y visuales de las detecciones, incluyendo gráficos y reportes en PDF.
- **Visualización en Tiempo Real**: Muestra las detecciones y seguimientos en tiempo real mientras procesa el video.

## Requisitos

### Lenguaje y Herramientas

- **Lenguaje**: Python 3.8 o superior
- **Librerías**:
  - [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
  - OpenCV
  - NumPy
  - Matplotlib
  - Seaborn
  - scikit-learn
  - SciPy
  - ReportLab
  - Pandas

### Instalación de Dependencias

Puedes instalar todas las dependencias necesarias utilizando el archivo `requirements.txt`. Asegúrate de tener `pip` instalado y ejecuta el siguiente comando en la raíz del proyecto:
(recomendable crear entorno virtual).

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```plaintext
vehicle_recognition/
├── data/
│   ├── input/               # Videos de entrada
│   │   └── video.mp4         # Video a procesar
│   └── output/              # Videos procesados
│       └── output.avi        # Video de salida con detecciones
├── reports/
│   ├── images/              # Gráficas generadas
│   │   ├── vehicle_counts.png
│   │   ├── confidence_distribution.png
│   │   └── vehicles_timeline.png
│   └
│   └── vehicle_analysis_report.pdf
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── analyzer.py      # Análisis y generación de reportes
│   │   ├── detector.py      # Detección de vehículos
│   │   ├── evaluator.py     # Evaluación del modelo
│   │   ├── sort.py          # Implementación del algoritmo SORT
│   │   └── tracker.py       # Rastreo de objetos
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py     # Constantes y configuraciones
│   │   ├── detection_utils.py  # Utilidades para detección
│   │   └── visualization.py # Funciones de visualización
│   └── __init__.py
├── docs/
│   └── informe_proyecto.docx # Documento de Word con el reporte del proyecto
├── .gitignore
├── README.md
├── requirements.txt
└── main.py                  # Punto de entrada principal
```

## Uso

### Ejecución del Proyecto

1. **Preparar el Video de Entrada**:
   
   Coloca el video que deseas procesar en la carpeta `data/input/` y asegúrate de que se llame `video.mp4`. Si el nombre es diferente, actualiza el nombre en el archivo `main.py`.

2. **Ejecutar el Detector y Generar Reporte**:

   Ejecuta el siguiente comando desde la raíz del proyecto:

   ```bash
   python main.py
   ```

   Este comando realizará las siguientes acciones:
   - Procesará el video de entrada mostrando las detecciones en tiempo real.
   - Guardará el video procesado en `data/output/output.avi`.
   - Generará gráficas en `reports/images/`.
   - Creará un reporte PDF con estadísticas y gráficas en `reports/pdf/vehicle_analysis_report.pdf`.


## Personalización

- **Ajuste de Umbrales y Configuraciones**:
  
  Puedes ajustar los umbrales de confianza y los colores de las clases en los archivos correspondientes:
  - `src/utils/constants.py`
  - `src/core/detector.py`

- **Cambio de Modelo YOLO**:
  
  El modelo YOLO utilizado por defecto es `yolov8m.pt`. Puedes cambiarlo por otro modelo (como `yolov8n.pt`) modificando la constante `MODEL_PATH` en `src/utils/constants.py`.

## Contribuciones

Si deseas contribuir a este proyecto, por favor sigue estos pasos:

1. Fork del repositorio.
2. Crea una nueva rama para tu feature (`git checkout -b feature/nueva-feature`).
3. Commit de tus cambios (`git commit -m 'Agregar nueva feature'`).
4. Push a la rama (`git push origin feature/nueva-feature`).
5. Abre un Pull Request.

## Licencia

Este proyecto está licenciado bajo la [MIT License](LICENSE).

## Agradecimientos

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) por su excelente librería de detección de objetos.
- [SORT Algorithm](https://github.com/abewley/sort) por su implementación de rastreo de objetos.
- Agradecimientos especiales a la comunidad de desarrolladores de Python y OpenCV por sus recursos y soporte.
