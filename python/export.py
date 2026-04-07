from ultralytics import YOLO

def pt_to_onnx():
    model = YOLO("/home/lalafua/Workspace/毕业设计/yolo-inference/models/best.pt")
    model.export(format = 'onnx')
    

def main():
    print("Start......")
    pt_to_onnx()
    print("End......")


if __name__ == "__main__":
    main()
