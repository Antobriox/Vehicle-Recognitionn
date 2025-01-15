def filter_detections_by_confidence(results, confidence_threshold=0.3):
    """
    Filtra las detecciones del modelo en función del umbral de confianza.
    """
    filtered_detections = []
    
    # Verificar si hay detecciones
    if not results[0].boxes:
        return filtered_detections
        
    for box in results[0].boxes:
        try:
            confidence = float(box.conf)
            class_id = int(box.cls)
            
            # Filtrar solo vehículos (car, motorcycle, bus, truck)
            if class_id not in [2, 3, 5, 7]:  # IDs de vehículos en COCO
                continue
                
            if confidence >= confidence_threshold:
                # Convertir a coordenadas x1,y1,x2,y2
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                filtered_detections.append({
                    'class_id': int(class_id),  # Asegurar que sea entero
                    'confidence': float(confidence),
                    'coordinates': (x1, y1, x2, y2)
                })
        except Exception as e:
            print(f"Error procesando detección: {e}")
            continue
            
    return filtered_detections 