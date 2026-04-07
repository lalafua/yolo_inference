from ultralytics import YOLO

def train_model():
    model = YOLO("/home/lalafua/Workspace/毕业设计/yolo-inference/models/yolo11n.pt")
    
    results = model.train(
        data = '/home/lalafua/Workspace/毕业设计/yolo-inference/dataset/data.yaml',
        epochs = 20,
        imgsz = 640,
        batch = 16,
        device = 'cpu',
        workers = 2,
        project = 'run',
        name = 'yolo_prj',
    )

    metrics = model.val()
    print(f"Metrics(mAP@50-95): {metrics.box.map}")

def main():
    print("Start......")
    train_model()
    print("End......")


if __name__ == "__main__":
    main()
