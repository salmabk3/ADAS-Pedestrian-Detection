# YOLO pedestrian detection module
from ultralytics import YOLO


class YoloDetector:
    def __init__(
        self,
        model_name="yolov8n.pt",
        conf_thres=0.4,        # seuil de confiance supprime les détections douteuses

#                   réduit fortement les faux positifs
        min_box_area=1500     # surface minimale d'une bounding box
    ):
        """
        Initialisation du détecteur YOLO pour les piétons
        """
        print(f"Chargement de YOLO ({model_name})...")
        self.model = YOLO(model_name)
        self.conf_thres = conf_thres
        self.min_box_area = min_box_area

    def detect(self, frame):
        """
        Détecte les piétons dans une image.

        Retour :
        Liste de boîtes [x1, y1, x2, y2, conf, class_id]
        """

        # Détection YOLO (classe 0 = person)
        results = self.model(
            frame,
            classes=[0], #garantit qu’on ne détecte QUE des personnes
            conf=self.conf_thres,
            verbose=False
        )

        boxes = results[0].boxes.data.tolist()
        filtered_boxes = []

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box

            # Calcul surface de la bounding box
            area = (x2 - x1) * (y2 - y1)

            # Filtrage par taille minimale
            if area >= self.min_box_area:
                filtered_boxes.append(box)

        return filtered_boxes
