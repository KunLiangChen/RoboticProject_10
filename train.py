from ultralytics import YOLO
if __name__ == '__main__':

    model = YOLO("yolo11n.pt")
    results = model.train(data="Minecraft_Capstone.v4-cow-creeper-sheep-pig-house-villager.yolov11/data.yaml",
                          epochs=100,
                          imgsz=640,
                          batch=0.6,
                          workers=4,
                          device="cuda:0",
                          )
    model.export(format="engine", dynamic=True)