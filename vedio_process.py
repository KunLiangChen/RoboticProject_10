import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo11x.pt")

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

# 创建视频写入器
def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    print(f"视频分辨率: {frame_width}x{frame_height}, 原始帧率: {fps} FPS")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer, frame_width, frame_height, fps

output_filename = "Minecraft_result.mp4"
video_path = r"Minecraft.mp4"

cap = cv2.VideoCapture(video_path)
writer, frame_width, frame_height, fps = create_video_writer(cap, output_filename)

prev_time = time.time()  # 记录初始时间

while True:
    success, img = cap.read()
    if not success:
        break

    # 计算运行 FPS
    start_time = time.time()

    # 识别 + 标注
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)

    end_time = time.time()
    fps_runtime = 1.0 / (end_time - start_time + 1e-8)  # 防止除零

    # 在画面上写真实运行 FPS
    cv2.putText(result_img, f"",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    height, width = img.shape[:2]
    cv2.putText(result_img, f"{frame_width}x{frame_height} Runtime FPS: {fps_runtime:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    writer.write(result_img)
    cv2.imshow("Image", result_img)

    if cv2.waitKey(1) & 0xFF == 27:  # 按 Esc 退出
        break

writer.release()
cap.release()
cv2.destroyAllWindows()