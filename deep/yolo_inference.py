from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.track('tennis_match2_cut.mp4',conf=0.2, save=True)

# print(result)
# print("boxes: ")

# for box in result[0].boxes:
#     print(box)