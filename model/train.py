from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # bigger model

model.train(
    data='/content/taka-2/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16 ) 