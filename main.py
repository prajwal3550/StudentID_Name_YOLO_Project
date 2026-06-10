from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")
image_path = "input_image.jpg"
results = model(image_path)
annotated_frame = results[0].plot()
cv2.imwrite("output_result.png", annotated_frame)
print("Detection completed. Output saved as output_result.png")
