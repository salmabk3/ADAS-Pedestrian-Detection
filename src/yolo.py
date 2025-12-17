# YOLO pedestrian detection module
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_name='yolov8n.pt'):
        # On charge le modèle (télécharge automatiquement si absent)
        print(f"Chargement de YOLO ({model_name})...")
        self.model = YOLO(model_name)
    
    def detect(self, frame):
        """
        Détecte les objets dans l'image.
        Retourne une liste de boîtes : [x1, y1, x2, y2, conf, class_id]
        """
        # classes=[0] force YOLO à ne chercher que les "Personnes"
        # verbose=False évite de polluer la console avec du texte
        results = self.model(frame, classes=[0], verbose=False)
        
        # On retourne les données des boîtes détectées
        return results[0].boxes.data.tolist()
