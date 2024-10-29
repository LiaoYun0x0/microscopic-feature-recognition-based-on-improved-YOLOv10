# 参考博客：https://blog.csdn.net/ljlqwer/article/details/129175087
from ultralytics import YOLO

model = YOLO("/first_disk/hongzheng/yolov10-104/runs/eight/exp/weights/best.pt")  # 权重地址

results = model.val(data="/first_disk/hongzheng/yolov10-main-v1/yolov10-main/ultralytics/cfg/datasets/eight.yaml", imgsz=640, batch=32, conf=0.001, iou=0.5, name='Tri', optimizer='Adam')  