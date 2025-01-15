import numpy as np

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        """
        Inicializa un objeto rastreado usando un filtro de Kalman.
        bbox: [x1, y1, x2, y2, score]
        """
        self.id = KalmanBoxTracker.count  # Asignar un ID único al objeto
        KalmanBoxTracker.count += 1  # Incrementar el contador global
        self.bbox = bbox[:4]  # Guardar las coordenadas del cuadro delimitador
        self.hits = 1  # Número de veces que el objeto ha sido rastreado con éxito
        self.age = 0  # Número de cuadros desde la última vez que el objeto fue actualizado
        self.time_since_update = 0  # Tiempo desde la última actualización

    def update(self, bbox):
        """
        Actualiza las coordenadas del cuadro delimitador del objeto.
        """
        self.bbox = bbox[:4]
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        """
        Predice la próxima posición del objeto rastreado.
        """
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def get_state(self):
        """
        Devuelve el estado actual del objeto (coordenadas y ID).
        """
        return (*self.bbox, self.id)


class Sort:
    def __init__(self, max_age=5, min_hits=2):
        """
        Inicializa el rastreador SORT.
        max_age: Número máximo de cuadros sin actualización antes de eliminar un objeto.
        min_hits: Número mínimo de hits antes de confirmar un objeto.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []  # Lista de objetos rastreados

    def update(self, detections):
        """
        Actualiza las trayectorias rastreadas usando nuevas detecciones.
        detections: Lista de detecciones en formato [x1, y1, x2, y2, score].
        """
        # Actualizar los objetos existentes
        for tracker in self.trackers:
            tracker.time_since_update += 1

        # Asociar las nuevas detecciones con los rastreadores existentes
        new_tracks = []
        for det in detections:
            new_tracker = KalmanBoxTracker(det)
            new_tracks.append(new_tracker)

        # Agregar los nuevos rastreadores a la lista
        self.trackers = new_tracks

        # Filtrar objetos rastreados basados en `max_age` y `min_hits`
        active_trackers = []
        for tracker in self.trackers:
            if tracker.time_since_update < self.max_age:
                active_trackers.append(tracker)
        self.trackers = active_trackers

        # Devolver el estado de los rastreadores activos
        results = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits or tracker.time_since_update == 0:
                results.append(tracker.get_state())

        return results
