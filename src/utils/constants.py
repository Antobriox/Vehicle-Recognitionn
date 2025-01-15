import os

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_PATH = os.path.join(DATA_DIR, 'input')
OUTPUT_PATH = os.path.join(DATA_DIR, 'output')
REPORTS_PATH = os.path.join(BASE_DIR, 'reports')

# Configuración de clases
CLASS_NAMES = {
    2: "Car",      
    3: "Motorcycle",  
    5: "Bus",       
    7: "Truck"      
}

CLASS_COLORS = {
    2: (255, 255, 255),  # Blanco para carros
    3: (0, 255, 0),      # Verde para motos
    5: (255, 255, 0),    # Amarillo para buses
    7: (255, 0, 0),      # Rojo para camiones
}

# Configuración del modelo
MODEL_PATH = 'yolov8m.pt' 