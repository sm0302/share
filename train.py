from ultralytics import YOLO

# YOLOv11 Segmentation 모델 불러오기 (사전 학습된 모델)
model = YOLO('yolo11n-seg.pt')

# 모델 훈련 + 파인튜닝 (초기 레이어 동결)
model.train(data='./coco_animals_yolo/data.yaml', epochs=50, imgsz=640, batch=16, freeze=10)
