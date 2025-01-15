import numpy as np

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.bbox = bbox[:4]
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def update(self, bbox):
        self.bbox = bbox[:4]
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def get_state(self):
        return (*self.bbox, self.id)

class Sort:
    def __init__(self, max_age=5, min_hits=2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []

    def update(self, detections):
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