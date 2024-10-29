import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov10-eight.yaml')
    model.train(data=r"/first_disk/hongzheng/yolov10-104/ultralytics/cfg/datasets/eight.yaml",
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  
                batch=16,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume=True, 
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/eight',
                name='exp',
                )

